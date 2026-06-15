"""
Training Utilities

Components:
- set_seed: Fix random seed
- CurriculumScheduler: Curriculum learning scheduler
- LinearWarmupCosineAnnealingLR: Learning rate scheduler
- Training and evaluation functions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import Dict, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class CurriculumScheduler:
    
    def __init__(self, total_epochs: int, transition_ratio: float = 0.5):
        self.total_epochs = total_epochs
        self.transition_epoch = int(total_epochs * transition_ratio)
        
        print(f"\n{'='*60}")
        print("Curriculum Learning Schedule:")
        print(f"  Phase 1.1 (Patch-level): Epoch 1 ~ {self.transition_epoch}")
        print(f"  Phase 1.2 (Pack-level):  Epoch {self.transition_epoch + 1} ~ {total_epochs}")
        print(f"{'='*60}\n")
    
    def get_masking_mode(self, epoch: int) -> str:
        if epoch < self.transition_epoch:
            return 'patch'
        else:
            return 'pack'
    
    def get_phase_name(self, epoch: int) -> str:
        if epoch < self.transition_epoch:
            return "Phase 1.1 (Patch-level)"
        else:
            return "Phase 1.2 (Pack-level)"


class LinearWarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        eta_min: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.cosine_epochs = total_epochs - warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            progress = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * progress
                for base_lr in self.base_lrs
            ]
        else:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            progress = cosine_epoch / self.cosine_epochs if self.cosine_epochs > 0 else 0
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def pretrain_epoch(
    model,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    mask_ratio: float = 0.75,
    masking_mode: str = 'patch',
    gradient_accumulation_steps: int = 1
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    accumulated_loss = 0.0
    
    desc = f"Pretraining ({masking_mode}-level)"
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=desc)):
        features, pack_coords = batch_data
        features = features.to(device)
        
        loss, pred, mask = model(features, mask_ratio, masking_mode=masking_mode, pack_coords=pack_coords)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        accumulated_loss += loss.item() * gradient_accumulation_steps
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += accumulated_loss
            accumulated_loss = 0.0
            num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    if accumulated_loss > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch(
    encoder,
    classifier,
    dataloader: DataLoader,
    task_names: List[str],
    task_info: Dict[str, Dict],
    optimizer: optim.Optimizer,
    criterion_dict: Dict[str, nn.Module],
    device: torch.device,
    freeze_encoder: bool = True,
    text_scaler=None
) -> Dict[str, float]:
    if freeze_encoder:
        encoder.eval()
    else:
        encoder.train()
    classifier.train()
    
    total_loss = 0.0
    task_losses = {task: 0.0 for task in task_names}
    num_batches = 0
    
    for features, labels_dict in tqdm(dataloader, desc="Training", leave=False):
        features = features.to(device)
        
        with torch.set_grad_enabled(not freeze_encoder):
            global_feat = encoder.get_representations(features, text_scaler=text_scaler)
        
        logits_dict = classifier(global_feat)
        
        batch_loss = 0.0
        for task in task_names:
            labels = labels_dict[task].to(device)
            
            if task_info[task]['task_type'] == 'binary':
                loss = criterion_dict[task](logits_dict[task].squeeze(-1), labels)
            else:
                loss = criterion_dict[task](logits_dict[task], labels)
            
            task_losses[task] += loss.item()
            batch_loss += loss
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_task_losses = {task: task_losses[task] / num_batches for task in task_names}
    
    return {'total': avg_loss, **avg_task_losses}


def evaluate(
    encoder,
    classifier,
    dataloader: DataLoader,
    task_names: List[str],
    task_info: Dict[str, Dict],
    device: torch.device,
    return_predictions: bool = False,
    text_scaler=None
) -> Dict:
    encoder.eval()
    classifier.eval()
    
    all_predictions = {task: [] for task in task_names}
    all_labels = {task: [] for task in task_names}
    
    with torch.no_grad():
        for features, labels_dict in tqdm(dataloader, desc="Evaluating", leave=False):
            features = features.to(device)
            
            global_feat = encoder.get_representations(features, text_scaler=text_scaler)
            logits_dict = classifier(global_feat)
            
            for task in task_names:
                labels = labels_dict[task].cpu().numpy()
                all_labels[task].extend(labels)
                
                if task_info[task]['task_type'] == 'binary':
                    probs = torch.sigmoid(logits_dict[task].squeeze(-1))
                    preds = (probs > 0.5).cpu().numpy().astype(int)
                else:
                    preds = torch.argmax(logits_dict[task], dim=-1).cpu().numpy()
                
                all_predictions[task].extend(preds)
    
    metrics = {}
    for task in task_names:
        acc = accuracy_score(all_labels[task], all_predictions[task])
        metrics[f'{task}_accuracy'] = acc
    
    if return_predictions:
        return metrics, all_predictions, all_labels
    return metrics

