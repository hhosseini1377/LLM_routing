from cpx_model.config import CPXTrainingConfig
import torch
from torch.utils.data import DataLoader
from cpx_model.cpx_causal_utils import TextRegressionDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_scheduler
from torch.amp import autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from cpx_model.cpx_causal_lm import CPXCausalLM
import time
import random
import numpy as np
from peft import LoraConfig

class CPXTrainer:
    def __init__(self, tokenizer, train_texts, train_labels, validation_texts, validation_labels, training_config):
        """
        Initialize CPX Trainer.
        
        Args:
            tokenizer: The tokenizer to use for text processing
            train_texts: List of training text samples
            train_labels: List of training labels
            validation_texts: List of validation text samples  
            validation_labels: List of validation labels
            training_config: CPXTrainingConfig instance with training parameters
        """
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.validation_texts = validation_texts
        self.validation_labels = validation_labels
        self.training_config = training_config
        self.world_size = torch.cuda.device_count()

    def setup(self, rank):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29505"
        
        # Set seeds BEFORE any model initialization
        torch.manual_seed(42)  # Same seed for all ranks!
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=self.world_size, rank=rank
        )

    def cleanup(self):
        dist.destroy_process_group()

    def preprocess_data(self, context_window, rank, batch_size):
        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True, seed=42)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)
        return loader

    def train(self, rank, batch_size, context_window, num_epochs, model_name):
        print(f'Training on rank {rank} started')

        self.setup(rank)

        # Get CPX token and ID
        cpx_token = self.training_config.cpx_token
        cpx_token_id = self.tokenizer.convert_tokens_to_ids(cpx_token)

        # Load model with CPX wrapper
        # Check if LoRA should be used (via config)
        use_lora = self.training_config.use_lora
        mask_lora_for_non_cpx = self.training_config.mask_lora_for_non_cpx

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            target_modules=self.training_config.lora_target_modules,
            lora_dropout=self.training_config.lora_dropout,
            bias=self.training_config.lora_bias,
            task_type=self.training_config.lora_task_type,
        )
        model = CPXCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cpx_token_id=cpx_token_id,
            num_labels=1,
            is_cpx_token_trainable=self.training_config.is_cpx_token_trainable,
            tokenizer_size=len(self.tokenizer),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable cache for training
            use_lora=use_lora,
            mask_lora_for_non_cpx=mask_lora_for_non_cpx,
            dropout_rate=self.training_config.dropout_rate,
            classifier_dropout=self.training_config.classifier_dropout,
            lora_config=lora_config,
            freeze_LoRA_layers=self.training_config.freeze_LoRA_layers,
            freeze_LoRA_start_layer_idx=self.training_config.freeze_LoRA_start_layer_idx
        ).to(rank)
        
        # Ensure cache is disabled (redundant but explicit)
        model.base_model.config.use_cache = False

        # Enable gradient checkpointing to reduce memory usage
        if self.training_config.gradient_checkpointing:
            # Enable on base model
            if hasattr(model.base_model, 'gradient_checkpointing_enable'):
                model.base_model.gradient_checkpointing_enable()
                print(f"Gradient checkpointing enabled on rank {rank}")
        else:
            print(f"Gradient checkpointing disabled on rank {rank}")
        
        ddp_model = DDP(model, device_ids=[rank])

        loader = self.preprocess_data(context_window, rank, batch_size)
        
        # Build optimizer parameter groups with optimized learning rates
        # Based on best practices for LoRA fine-tuning and complexity classification
        param_groups = [
            # Classifier: New layer, can handle moderate LR
            {"params": model.classifier.parameters(), 
             "lr": self.training_config.classifier_lr, 
             "weight_decay": self.training_config.weight_decay},
        ]
        
        # Add embedding parameters if trainable
        if self.training_config.is_cpx_token_trainable:
            if model.use_lora:
                embedding_layer = model.base_model.get_base_model().get_input_embeddings()
            else:
                embedding_layer = model.base_model.get_input_embeddings()
            # CPX Embedding: Single token, needs careful/slower tuning
            param_groups.append({
                "params": embedding_layer.parameters(), 
                "lr": self.training_config.embedding_lr,
                "weight_decay": self.training_config.embedding_weight_decay
            })
        
        # Add LoRA parameters if using LoRA
        if model.use_lora:
            lora_params = [p for n, p in model.base_model.named_parameters() if 'lora_' in n and p.requires_grad]
            if len(lora_params) > 0:
                # LoRA: Standard LoRA fine-tuning LR
                param_groups.append({
                    "params": lora_params, 
                    "lr": self.training_config.lora_lr,
                    "weight_decay": self.training_config.weight_decay
                })
                print(f"  ✓ Optimizer includes {len(lora_params)} LoRA parameter groups")
        
        # Print learning rate configuration
        if rank == 0:
            print(f"  Learning Rates:")
            print(f"    - Classifier: {self.training_config.classifier_lr}")
            if self.training_config.is_cpx_token_trainable:
                print(f"    - CPX Embedding: {self.training_config.embedding_lr}")
            if model.use_lora:
                print(f"    - LoRA Adapters: {self.training_config.lora_lr}")
        
        optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

        if self.training_config.scheduler == "linear":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * self.training_config.warmup_steps)    

            scheduler = get_scheduler(
                name=self.training_config.scheduler,
                optimizer=optimizer,    
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.training_config.scheduler == "cosine":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * self.training_config.warmup_steps)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.training_config.scheduler == "ReduceLROnPlateau":
            if self.training_config.METRIC == "f1":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
            elif self.training_config.METRIC == "loss":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {self.training_config.scheduler}")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        criterion = nn.BCEWithLogitsLoss()
        log_path = f"{self.training_config.LOG_DIR}/log_cpx_{timestamp}.txt"
        
        ddp_model.train()
        
        if self.training_config.METRIC == "f1":
            best_score = 0
        elif self.training_config.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
        metric = self.training_config.METRIC
        # Write the setup to the log file 
        if rank == 0:
            with open(log_path, "a") as f:
                f.write(
                    f"model: {model_name}, \n"
                    f"dataset: {self.training_config.dataset}, \n"
                    f"use_lora: {use_lora}, \n"
                    f'mask_lora_for_non_cpx: {mask_lora_for_non_cpx}, \n'
                    f"metric: {self.training_config.METRIC}, \n"
                    f"batch_size: {batch_size*self.world_size}, \n"
                    f"context_window: {context_window}, \n"
                    f"train_size: {len(self.train_texts)}, \n"
                    f"gradient_checkpointing: {self.training_config.gradient_checkpointing}\n"
                    f'classifier_lr: {self.training_config.classifier_lr}, \n'
                    f'embedding_lr: {self.training_config.embedding_lr}, \n'
                    f'lora_lr: {self.training_config.lora_lr}, \n'
                    f'weight_decay: {self.training_config.weight_decay}, \n'
                    f'embedding_weight_decay: {self.training_config.embedding_weight_decay}, \n'
                    f'scheduler: {self.training_config.scheduler}, \n'
                    f'warmup_steps: {self.training_config.warmup_steps}, \n'
                    f'patience: {self.training_config.patience}, \n'
                    f'max_grad_norm: {self.training_config.max_grad_norm}, \n'
                    f'dropout_rate: {self.training_config.dropout_rate}, \n'
                    f'classifier_dropout: {self.training_config.classifier_dropout}, \n'
                    f'lora_r: {self.training_config.lora_r}, \n'
                    f'lora_alpha: {self.training_config.lora_alpha}, \n'
                    f'lora_dropout: {self.training_config.lora_dropout}, \n'
                    f'lora_target_modules: {self.training_config.lora_target_modules}, \n'
                    f'lora_bias: {self.training_config.lora_bias}, \n'
                    f'lora_task_type: {self.training_config.lora_task_type}, \n'
                    f'freeze_LoRA_layers: {self.training_config.freeze_LoRA_layers}, \n'
                    f'freeze_LoRA_start_layer_idx: {self.training_config.freeze_LoRA_start_layer_idx}, \n'
                )

        # Evaluate the model at start
        if metric == "f1":
            per_gpu_evaluation_batch_size = self.training_config.evaluation_batch_size // self.world_size
            score, accuracy, best_threshold = self.evaluate_with_optimal_threshold_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
            if rank == 0:
                score_str = f"Avg F1 score on the validation set: {score:.4f}, Avg Accuracy on the validation set: {accuracy:.4f}, Best threshold: {best_threshold:.4f}"
            else:
                score_str = "Evaluation completed on other ranks"
        elif metric == "loss":
            score = self.evaluate_flat(self.training_config.evaluation_batch_size, context_window)
            score_str = f"Avg Binary Cross Entropy Loss on the validation set: {score:.4f}"
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")

        # Synchronize all processes before proceeding
        dist.barrier(device_ids=[rank])

        # Log the results
        if rank == 0:
            print(f"At start: {score_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"At start: {score_str}\n"
                )

        for epoch in range(num_epochs):
            if rank == 0:
                print(f'Epoch {epoch + 1} started')
            dist.barrier(device_ids=[rank])
            total_loss = 0
            loader.sampler.set_epoch(epoch)

            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)
                optimizer.zero_grad()  
            
                with autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, targets)

                loss.backward()
                
                # Apply gradient clipping
                if self.training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), self.training_config.max_grad_norm)
                
                optimizer.step()

                total_loss += loss.item()
                if self.training_config.scheduler in ["linear", "cosine"]:
                    scheduler.step()

            loss_tensor = torch.tensor(total_loss, device=rank)
            count_tensor = torch.tensor(len(loader), device=rank, dtype=torch.float)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            train_loss = loss_tensor.item() / count_tensor.item()

            # Evaluate the model
            if metric == "f1":
                per_gpu_evaluation_batch_size = self.training_config.evaluation_batch_size // self.world_size
                score, accuracy, best_threshold = self.evaluate_with_optimal_threshold_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
                if rank == 0:
                    score_str = f"Avg F1 score on the validation set: {score:.4f}, Avg Accuracy on the validation set: {accuracy:.4f}, Best threshold: {best_threshold:.4f}"
                else:
                    score_str = "Evaluation completed on other ranks"
            elif metric == "loss":
                score = self.evaluate_flat(self.training_config.evaluation_batch_size, context_window)
                score_str = f"Avg Binary Cross Entropy Loss on the validation set: {score:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            # Synchronize all processes before proceeding
            dist.barrier(device_ids=[rank])
        
            if self.training_config.scheduler == "ReduceLROnPlateau":
                if rank == 0 and score is not None:
                    scheduler.step(score)

            # Log the results
            if rank == 0:
                print(f"Epoch {epoch + 1}, {score_str}")
                with open(log_path, "a") as f:
                    f.write(
                        f"Epoch {epoch + 1}, Avg Binary Cross Entropy Loss on the training set: {train_loss:.4f}, {score_str}\n"
                    )

            # local flag: only rank 0 decides
            if rank == 0 and score is not None:
                if self.training_config.METRIC == "f1":
                    comparison = score > best_score
                elif self.training_config.METRIC == "loss":
                    comparison = score < best_score
                else:
                    raise ValueError(f"Unsupported metric: {self.training_config.METRIC}")

                if comparison:
                    best_score = score
                    best_model_state = ddp_model.module.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
            else:
                comparison = False  # other ranks don't evaluate

            # rank 0 decides whether to stop
            if rank == 0 and patience_counter >= patience:
                stop_flag = torch.tensor([1.0], device=rank)
            else:
                stop_flag = torch.tensor([0.0], device=rank)

            # share stop_flag with everyone
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

            # check and break
            if stop_flag.item() == 1.0:
                if rank == 0:
                    print("⏹️ Early stopping triggered!")
                break

        self.cleanup()
        if rank == 0 and best_model_state is not None:
            save_directory = self.training_config.MODEL_DIR
            os.makedirs(save_directory, exist_ok=True)
            model_basename = model_name.replace('/', '_')
            torch.save(best_model_state, f"{save_directory}/model_{model_basename}_cpx_{timestamp}.pth")
            print(f"Model saved to {save_directory}/model_{model_basename}_cpx_{timestamp}.pth")

    def run(self, batch_size, context_window, num_epochs, model_name):
        try:
            mp.spawn(self.train, args=(batch_size, context_window, num_epochs, model_name), nprocs=self.world_size)
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup()
            raise e

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cuda")  # or "cuda"
        self.model.load_state_dict(state_dict)

    def evaluate_flat(self, batch_size, context_window,):
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].float().to(self.model.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")
        self.model.train()
        return total_loss / len(loader)

    def get_validation_probabilities_distributed(self, ddp_model, rank, batch_size, context_window):
        """Get probabilities for validation set in one forward pass - distributed version"""
        ddp_model.eval()
        dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)
                with autocast('cuda', dtype=torch.bfloat16):                
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
                all_targets.append(targets.int())

        ddp_model.train()
        
        # Concatenate as tensors
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Pad to same size if dataset shards are unequal
        local_size = torch.tensor([all_probs_tensor.size(0)], device=rank)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        pad_size = max_size - all_probs_tensor.size(0)
        if pad_size > 0:
            all_probs_tensor = torch.cat([all_probs_tensor, torch.zeros(pad_size, *all_probs_tensor.shape[1:], device=all_probs_tensor.device, dtype=all_probs_tensor.dtype)])
            all_targets_tensor = torch.cat([all_targets_tensor, torch.zeros(pad_size, *all_targets_tensor.shape[1:], device=all_targets_tensor.device, dtype=all_targets_tensor.dtype)])

        # Allocate gather buffers
        gathered_probs = [torch.zeros_like(all_probs_tensor) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]

        # Gather from all ranks
        dist.all_gather(gathered_probs, all_probs_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)

        # Only return results on rank 0
        if rank == 0:
            # Concatenate all gathered results
            all_probs_global = torch.cat(gathered_probs, dim=0).to(torch.float32)
            all_targets_global = torch.cat(gathered_targets, dim=0)
            return all_probs_global.view(-1).cpu().numpy(), all_targets_global.view(-1).cpu().numpy()
        else:
            return None, None

    def find_best_threshold(self, probs, targets, threshold_range=(0.1, 0.9), num_thresholds=50):
        """Find the best threshold by evaluating F1 scores for different thresholds"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        best_f1 = 0
        best_threshold = 0.5
        best_accuracy = 0
        
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            f1 = f1_score(targets, preds, average='macro')
            accuracy = accuracy_score(targets, preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = accuracy
                
        return best_f1, best_accuracy, best_threshold

    def evaluate_accuracy_distributed(self, ddp_model, rank, batch_size, context_window, threshold=0.5):
        ddp_model.eval()
        """Distributed version of evaluate_accuracy for multi-GPU training"""
        dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)
                with autocast('cuda', dtype=torch.bfloat16):                
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

                all_preds.append(preds)
                all_targets.append(targets.int())

        ddp_model.train()
        
        # Concatenate as tensors (stay on CPU unless needed on GPU)
        all_preds_tensor = torch.cat(all_preds, dim=0)
        
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Pad to same size if dataset shards are unequal
        local_size = torch.tensor([all_preds_tensor.size(0)], device=rank)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        pad_size = max_size - all_preds_tensor.size(0)
        if pad_size > 0:
            all_preds_tensor = torch.cat([all_preds_tensor, torch.zeros(pad_size, *all_preds_tensor.shape[1:], device=all_preds_tensor.device, dtype=all_preds_tensor.dtype)])
            all_targets_tensor = torch.cat([all_targets_tensor, torch.zeros(pad_size, *all_targets_tensor.shape[1:], device=all_targets_tensor.device, dtype=all_targets_tensor.dtype)])

        # Allocate gather buffers
        gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]

        # Gather from all ranks
        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)

        # Only compute metrics on rank 0
        if rank == 0:
            # Concatenate all gathered results
            all_preds_global = torch.cat(gathered_preds, dim=0) 
            all_targets_global = torch.cat(gathered_targets, dim=0)
            y_pred = all_preds_global.view(-1).cpu().numpy()
            y_true = all_targets_global.view(-1).cpu().numpy()
            
            # Compute macro F1 score
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            return macro_f1, accuracy
        else:
            return None, None

    def evaluate_with_optimal_threshold_distributed(self, ddp_model, rank, batch_size, context_window):
        """Evaluate using optimal threshold found by validationing multiple thresholds"""
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities_distributed(ddp_model, rank, batch_size, context_window)
        
        if rank == 0 and probs is not None:
            # Find best threshold
            best_f1, best_accuracy, best_threshold = self.find_best_threshold(probs, targets)
            return best_f1, best_accuracy, best_threshold
        else:
            return None, None, None

    def get_validation_probabilities(self, batch_size, context_window):
        """Get probabilities for validation set in one forward pass - single GPU version"""
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_targets.append(targets.int().cpu())

        self.model.train()
        
        # Concatenate all probabilities and targets
        all_probs = torch.cat(all_probs, dim=0).to(torch.float32).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
        return all_probs.flatten(), all_targets.flatten()

    def evaluate_with_optimal_threshold(self, batch_size, context_window):
        """Evaluate using optimal threshold found by validationing multiple thresholds - single GPU version"""
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities(batch_size, context_window)
        
        # Find best threshold
        best_f1, best_accuracy, best_threshold = self.find_best_threshold(probs, targets)
        return best_f1, best_accuracy, best_threshold

    def evaluate_accuracy(self, batch_size, context_window, threshold=0.5):
        """Single GPU version - kept for backward compatibility"""
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

                all_preds.append(preds.cpu())
                all_targets.append(targets.int().cpu())

        self.model.train()
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        # Compute macro F1 score
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())
        return macro_f1, accuracy