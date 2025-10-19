from cpx_model.config import CPXTrainingConfig, MistralTrainingConfig
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
from cpx_model.cpx_causal_config import CPXConfig
import time
import random
import numpy as np

class CPXTrainer:
    def __init__(self, tokenizer, train_texts=None, train_labels=None, test_texts=None, test_labels=None):
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.world_size = torch.cuda.device_count()

    def setup(self, rank):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        
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

    def train(self, rank, batch_size, context_window, num_epochs, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        print(f'Training on rank {rank} started')

        self.setup(rank)

        # Get CPX token and ID
        cpx_token = CPXTrainingConfig.cpx_token
        cpx_token_id = self.tokenizer.convert_tokens_to_ids(cpx_token)

        # Load model with CPX wrapper
        model = CPXCausalLM.from_pretrained(
            model_name,
            cpx_token_id=cpx_token_id,
            num_labels=1,
            is_cpx_token_trainable=CPXTrainingConfig.is_cpx_token_trainable,
            tokenizer_size=len(self.tokenizer),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False  # Disable cache for training
        ).to(rank)
        
        # Ensure cache is disabled (redundant but explicit)
        model.base_model.config.use_cache = False

        # Enable gradient checkpointing to reduce memory usage
        if CPXTrainingConfig.gradient_checkpointing:
            # Enable on base model
            if hasattr(model.base_model, 'gradient_checkpointing_enable'):
                model.base_model.gradient_checkpointing_enable()
                print(f"Gradient checkpointing enabled on rank {rank}")
        else:
            print(f"Gradient checkpointing disabled on rank {rank}")
        
        ddp_model = DDP(model, device_ids=[rank])

        loader = self.preprocess_data(context_window, rank, batch_size)
        
        optimizer = AdamW([
            {"params": model.classifier.parameters(), "lr": 1e-3, "weight_decay": 0.01},
            {"params": [model.base_model.get_input_embeddings().weight], "lr": 2e-3, "weight_decay": 0.0},
        ], betas=(0.9, 0.999), eps=1e-8)

        if CPXTrainingConfig.scheduler == "linear":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * CPXTrainingConfig.warmup_steps)    

            scheduler = get_scheduler(
                name=CPXTrainingConfig.scheduler,
                optimizer=optimizer,    
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif CPXTrainingConfig.scheduler == "ReduceLROnPlateau":
            if CPXTrainingConfig.METRIC == "f1":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
            elif CPXTrainingConfig.METRIC == "loss":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {CPXTrainingConfig.scheduler}")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        criterion = nn.BCEWithLogitsLoss()
        log_path = f"{CPXTrainingConfig.LOG_DIR}/log_cpx_{timestamp}.txt"
        
        ddp_model.train()
        
        if CPXTrainingConfig.METRIC == "f1":
            best_score = 0
        elif CPXTrainingConfig.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
        metric = CPXTrainingConfig.METRIC
        # Write the setup to the log file 
        if rank == 0:
            with open(log_path, "a") as f:
                f.write(
                    f"model: {model_name}, "
                    f"metric: {CPXTrainingConfig.METRIC}, "
                    f"batch_size: {batch_size*self.world_size}, "
                    f"context_window: {context_window}, "
                    f"train_size: {len(self.train_texts)}, "
                    f"gradient_checkpointing: {CPXTrainingConfig.gradient_checkpointing}\n"
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
                optimizer.step()

                total_loss += loss.item()

                if CPXTrainingConfig.scheduler == "linear":
                    scheduler.step()

            loss_tensor = torch.tensor(total_loss, device=rank)
            count_tensor = torch.tensor(len(loader), device=rank, dtype=torch.float)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            train_loss = loss_tensor.item() / count_tensor.item()

            # Evaluate the model
            if metric == "f1":
                per_gpu_evaluation_batch_size = CPXTrainingConfig.evaluation_batch_size // self.world_size
                score, accuracy = self.evaluate_accuracy_distributed(ddp_model, rank, per_gpu_evaluation_batch_size, context_window)
                if rank == 0:
                    score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {accuracy:.4f}"
                else:
                    score_str = "Evaluation completed on other ranks"
            elif metric == "loss":
                score = self.evaluate_flat(MistralTrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            # Synchronize all processes before proceeding
            dist.barrier(device_ids=[rank])

            # Lines 173-179 - ONLY rank 0 should step the scheduler
            if CPXTrainingConfig.scheduler == "ReduceLROnPlateau":
                if rank == 0 and score is not None:
                    scheduler.step(score)

            # Log the results
            if rank == 0:
                print(f"Epoch {epoch + 1}, {score_str}")
                with open(log_path, "a") as f:
                    f.write(
                        f"Epoch {epoch + 1}, Avg Loss on the training set: {train_loss:.4f}, {score_str}\n"
                    )

            # local flag: only rank 0 decides
            if rank == 0 and score is not None:
                if CPXTrainingConfig.METRIC == "f1":
                    comparison = score > best_score
                elif CPXTrainingConfig.METRIC == "loss":
                    comparison = score < best_score
                else:
                    raise ValueError(f"Unsupported metric: {CPXTrainingConfig.METRIC}")

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
            save_directory = CPXTrainingConfig.MODEL_DIR
            os.makedirs(save_directory, exist_ok=True)
            model_basename = model_name.replace('/', '_')
            torch.save(best_model_state, f"{save_directory}/model_{model_basename}_cpx_{timestamp}.pth")
            print(f"Model saved to {save_directory}/model_{model_basename}_cpx_{timestamp}.pth")

    def run(self, batch_size, context_window, num_epochs, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
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
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

    def evaluate_accuracy_distributed(self, ddp_model, rank, batch_size, context_window):
        ddp_model.eval()
        """Distributed version of evaluate_accuracy for multi-GPU training"""
        dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
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

                # Apply sigmoid and threshold at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

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

    def evaluate_accuracy(self, batch_size, context_window,):
        """Single GPU version - kept for backward compatibility"""
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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

                # Apply sigmoid and threshold at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

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


# Backward compatibility alias
MistralTrainer = CPXTrainer