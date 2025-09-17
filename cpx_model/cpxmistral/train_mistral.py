from cpx_model.cpxmistral.config import MistralTrainingConfig
import torch
from torch.utils.data import DataLoader
from cpx_model.cpxmistral.config import MistralTrainingConfig
from cpx_model.cpxmistral.utils import TextRegressionDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_scheduler
from torch.amp import GradScaler, autocast
import os
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from cpx_model.cpxmistral.cpx_mistral import MyMistral
from cpx_model.cpxmistral.cpxmistralconfig import CPXMistralConfig
class MistralTrainer:
    def __init__(self, tokenizer, train_texts=None, train_labels=None, test_texts=None, test_labels=None):
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.world_size = torch.cuda.device_count()

    def setup(self, rank):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=self.world_size, rank=rank
        )
        torch.cuda.set_device(rank)

    def cleanup(self):
        dist.destroy_process_group()

    def preprocess_data(self, context_window, rank, batch_size):
        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)
        return loader

    def train(self, rank, batch_size, context_window, num_epochs):
        print(f'training on rank {rank}')
        self.setup(rank)

        mistral_config = CPXMistralConfig.from_pretrained(pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", num_labels=1, cpx_token=MistralTrainingConfig.cpx_token)
        mistral_config.tokenizer_size = len(self.tokenizer)
        mistral_config.cpx_token_id = self.tokenizer.convert_tokens_to_ids(MistralTrainingConfig.cpx_token)
        model = MyMistral.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.1",
            config=mistral_config,
            torch_dtype=torch.bfloat16,  # Use bfloat16
            low_cpu_mem_usage=True      # Reduce CPU memory usage during loading
        ).to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        loader = self.preprocess_data(context_window, rank, batch_size)
        
        learning_rate = MistralTrainingConfig.learning_rate 
        weight_decay = MistralTrainingConfig.weight_decay
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if MistralTrainingConfig.scheduler == "linear":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * MistralTrainingConfig.warmup_steps)    

            scheduler = get_scheduler(
                name=MistralTrainingConfig.scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif MistralTrainingConfig.scheduler == "ReduceLROnPlateau":
            if MistralTrainingConfig.METRIC == "f1":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
            elif MistralTrainingConfig.METRIC == "loss":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {MistralTrainingConfig.scheduler}")

        criterion = nn.BCEWithLogitsLoss()
        log_path = f"{MistralTrainingConfig.LOG_DIR}/log_mistral.txt"
        
        ddp_model.train()
        
        if MistralTrainingConfig.METRIC == "f1":
            best_score = 0
        elif MistralTrainingConfig.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
        metric = MistralTrainingConfig.METRIC

        # Write the setup to the log file incudling 
        with open(log_path, "a") as f:
            f.write(f"metric: {MistralTrainingConfig.METRIC}, "
                   f"batch_size: {batch_size}, "
                   f"context_window: {context_window}, "
                   f"train_size: {len(self.train_texts)}, "
                   f"dropout: {MistralTrainingConfig.dropout_rate}, "
                   f"layers_to_freeze: {MistralTrainingConfig.layers_to_freeze}, "
                   f"freeze_layers: {MistralTrainingConfig.freeze_layers}, "
                   f"classifier_dropout: {MistralTrainingConfig.classifier_dropout}, "
                   f"learning_rate: {MistralTrainingConfig.learning_rate}"
                   f"weight_decay: {MistralTrainingConfig.weight_decay}\n")

        for epoch in range(num_epochs):
            total_loss = 0
            loader.sampler.set_epoch(epoch)
            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)

                optimizer.zero_grad()  
            
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = ddp_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")

                if MistralTrainingConfig.scheduler == "linear":
                    scheduler.step()

            train_loss = total_loss / len(loader)
            
            # Check for NaN weights after each epoch
            has_nan_weights = False
            for name, param in ddp_model.named_parameters():
                if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"Warning: NaN/Inf detected in {name} after epoch {epoch + 1}")
                    has_nan_weights = True
            
            if has_nan_weights:
                print("Stopping training due to NaN weights")
                break

            # Evaluate the model
            if metric == "f1":
                score, accuracy_score = self.evaluate_accuracy(MistralTrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {accuracy_score:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(MistralTrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            if MistralTrainingConfig.scheduler == "ReduceLROnPlateau":
                scheduler.step(score)

            # Log the results
            print(f"Epoch {epoch + 1}, {score_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}, Avg Loss on the training set: {train_loss:.4f}, {score_str}\n"
                )

            # Select metric and direction
            if MistralTrainingConfig.METRIC == "f1":
                comparison = score > best_score
            elif MistralTrainingConfig.METRIC == "loss":
                comparison = score < best_score
            else:
                raise ValueError(f"Unsupported metric: {MistralTrainingConfig.METRIC}")

            # Early stopping logic
            if comparison:
                best_score = score
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered!")
                    break
        self.cleanup()
        save_directory = MistralTrainingConfig.MODEL_DIR
        torch.save(best_model_state, f"{save_directory}/model_{self.model_name}_{self.pooling_strategy}.pth")

    def run(self, batch_size, context_window, num_epochs):
        mp.spawn(self.train, args=(batch_size, context_window, num_epochs), nprocs=self.world_size)
          
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

    def evaluate_accuracy(self, batch_size, context_window,):
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
        
    

