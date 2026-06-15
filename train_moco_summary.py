"""
Summary Token Level MoCo Contrastive Learning

Contrastive learning at summary token level.
Positive sample strategies:
  1. adjacent: Use neighboring summary pack as positive sample
  2. noise: Use self as positive sample with Gaussian noise added before encoder
  3. mix: Randomly choose adjacent or noise strategy per sample
Negative samples: Summary tokens from other slides (avoid nearby files)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import argparse
import math
import copy
import random

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False

from network import HSANConfig
from model import HSANForMAE
from utils import set_seed


class MoCoSummaryToken(nn.Module):
    """Summary Token Level MoCo"""
    
    def __init__(self, base_encoder, embed_dim=512, proj_dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.embed_dim = embed_dim
        
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(base_encoder)
        
        # Summary token projector
        self.projector_q = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim)
        )
        self.projector_k = copy.deepcopy(self.projector_q)
        
        # Freeze key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        # Queue for negative samples
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Enqueue multiple keys (summary tokens)"""
        batch_size = keys.shape[0]
        if batch_size == 0:
            return
            
        ptr = int(self.queue_ptr)
        
        # Handle wrap-around
        if ptr + batch_size > self.K:
            first_part = self.K - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def encode_summary_tokens(self, x, encoder, projector, summary_indices):
        """Encode and get projected summary tokens"""
        encoded = encoder.encoder(x)  # (1, seq_len, embed_dim)
        summary_tokens = encoded[:, summary_indices, :]  # (1, num_summaries, embed_dim)
        summary_tokens = summary_tokens.squeeze(0)  # (num_summaries, embed_dim)
        projected = projector(summary_tokens)  # (num_summaries, proj_dim)
        return F.normalize(projected, dim=1)

    def forward(self, x_q, x_k, summary_indices_q, summary_indices_k, pair_indices):
        """
        Args:
            x_q: query input (1, seq_len, embed_dim)
            x_k: key input (1, seq_len, embed_dim) - 可能是加噪版本或同一个输入
            summary_indices_q: query 的 summary token 索引
            summary_indices_k: key 的 summary token 索引
            pair_indices: list of (q_idx, k_idx) pairs - 正样本对索引
        """
        # Get query summary tokens
        q_all = self.encode_summary_tokens(x_q, self.encoder_q, self.projector_q, summary_indices_q)
        
        with torch.no_grad():
            self._momentum_update()
            k_all = self.encode_summary_tokens(x_k, self.encoder_k, self.projector_k, summary_indices_k)
        
        # Extract positive pairs
        q_indices = [p[0] for p in pair_indices]
        k_indices = [p[1] for p in pair_indices]
        
        q = q_all[q_indices]  # (num_pairs, proj_dim)
        k = k_all[k_indices]  # (num_pairs, proj_dim)
        
        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (num_pairs, 1)
        
        # Negative logits (from queue)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (num_pairs, K)
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # Enqueue all key summary tokens
        self._dequeue_and_enqueue(k_all)
        
        return logits, labels


class MoCoSummaryDataset(Dataset):
    
    def __init__(self, feature_dir, min_gap=10):
        self.feature_dir = feature_dir
        self.file_list = sorted([f for f in os.listdir(feature_dir) if f.endswith('.pt')])
        self.min_gap = min_gap
        print(f"Found {len(self.file_list)} samples for MoCo training")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filepath = os.path.join(self.feature_dir, self.file_list[idx])
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        return data['features'], idx
    
    def get_distant_sample(self, current_idx):
        n = len(self.file_list)
        if n <= 2 * self.min_gap:
            candidates = [i for i in range(n) if i != current_idx]
        else:
            candidates = [i for i in range(n) if abs(i - current_idx) >= self.min_gap]
        
        if len(candidates) == 0:
            candidates = [i for i in range(n) if i != current_idx]
        
        return random.choice(candidates)


def get_summary_indices(seq_len, pack_size):
    block_stride = pack_size + 1
    num_packs = math.ceil((seq_len - 1) / block_stride)
    indices = []
    for k in range(num_packs):
        summ_idx = min(1 + (k + 1) * block_stride - 1, seq_len - 1)
        indices.append(summ_idx)
    return torch.tensor(indices, dtype=torch.long)


def get_adjacent_pairs(num_summaries, num_pairs=None):
    if num_summaries < 2:
        return []
    
    pairs = []
    for i in range(num_summaries - 1):
        pairs.append((i, i + 1))
        pairs.append((i + 1, i))
    
    if num_pairs is not None and len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)
    
    return pairs


def get_noise_pairs(num_summaries, num_pairs=None):
    pairs = [(i, i) for i in range(num_summaries)]
    
    if num_pairs is not None and len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)
    
    return pairs


def add_gaussian_noise(features, noise_std=0.1):
    noise = torch.randn_like(features) * noise_std
    return features + noise


def drop_packs(features, pack_size, drop_ratio, max_packs=800):
    seq_len = features.size(0)
    block_stride = pack_size + 1
    num_packs = math.ceil((seq_len - 1) / block_stride)
    
    if num_packs <= 1:
        return features
    
    num_keep = max(1, int(num_packs * (1 - drop_ratio)))
    num_keep = min(num_keep, max_packs)
    
    keep_packs = torch.randperm(num_packs)[:num_keep].sort()[0]
    
    keep_indices = [0]
    for pack_idx in keep_packs:
        start = 1 + pack_idx * block_stride
        end = min(start + block_stride, seq_len)
        keep_indices.extend(range(start, end))
    
    return features[keep_indices]


def main():
    parser = argparse.ArgumentParser(description="Summary Token Level MoCo")
    
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs_moco_summary")
    parser.add_argument("--pretrained-checkpoint", type=str, required=True)
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--projector-lr", type=float, default=1e-4)
    parser.add_argument("--drop-ratio", type=float, default=0.2)
    parser.add_argument("--max-packs", type=int, default=800)
    
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--queue-size", type=int, default=65536,
                        help="Queue size for negative samples")
    parser.add_argument("--momentum", type=float, default=0.999)
    parser.add_argument("--temperature", type=float, default=0.07)
    
    parser.add_argument("--positive-mode", type=str, default="adjacent",
                        choices=["adjacent", "noise", "mix"],
                        help="Positive sample strategy: adjacent (neighboring pack), noise (self + gaussian), or mix (random choice)")
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Standard deviation of Gaussian noise (for noise mode)")
    parser.add_argument("--max-pairs-per-sample", type=int, default=50,
                        help="Maximum number of positive pairs per sample")
    parser.add_argument("--min-file-gap", type=int, default=10,
                        help="Minimum file index gap to avoid same patient")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    run = None
    if SWANLAB_AVAILABLE:
        run = swanlab.init(
            project="HSAN-MoCo-Summary",
            experiment_name=os.path.basename(args.output_dir),
            config=vars(args),
            logdir=os.path.join(args.output_dir, "swanlab_logs")
        )
    
    print(f"Loading pretrained model from {args.pretrained_checkpoint}")
    checkpoint = torch.load(args.pretrained_checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', HSANConfig())
    
    encoder = HSANForMAE(config, use_gradient_checkpointing=True).to(device)
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = MoCoSummaryToken(
        base_encoder=encoder,
        embed_dim=config.embed_dim,
        proj_dim=args.proj_dim,
        K=args.queue_size,
        m=args.momentum,
        T=args.temperature
    ).to(device)
    
    optimizer = optim.AdamW([
        {'params': model.encoder_q.parameters(), 'lr': args.encoder_lr},
        {'params': model.projector_q.parameters(), 'lr': args.projector_lr}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    dataset = MoCoSummaryDataset(args.feature_dir, min_gap=args.min_file_gap)
    
    print(f"\nStarting Summary Token Level MoCo training for {args.epochs} epochs...")
    print(f"Positive mode: {args.positive_mode}")
    if args.positive_mode in ["noise", "mix"]:
        print(f"Noise std: {args.noise_std}")
    print(f"Drop ratio: {args.drop_ratio}, Max packs: {args.max_packs}")
    print(f"Queue size: {args.queue_size}, Min file gap: {args.min_file_gap}")
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        total_pairs = 0
        
        mode_counts = {"adjacent": 0, "noise": 0}
        
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for idx in indices:
            features, file_idx = dataset[idx]
            
            if features.size(0) > 200000:
                print(f"Skipping large sample {idx}: {features.size(0)} tokens")
                continue
            
            features = drop_packs(features, config.pack_size, args.drop_ratio, args.max_packs)
            
            seq_len = features.size(0)
            summary_indices = get_summary_indices(seq_len, config.pack_size).to(device)
            num_summaries = len(summary_indices)
            
            if num_summaries < 2:
                continue
            
            if args.positive_mode == "mix":
                current_mode = random.choice(["adjacent", "noise"])
                mode_counts[current_mode] += 1
            else:
                current_mode = args.positive_mode
            
            if current_mode == "adjacent":
                x_q = features.unsqueeze(0).to(device)
                x_k = features.unsqueeze(0).to(device)
                summary_indices_q = summary_indices
                summary_indices_k = summary_indices
                
                pair_indices = get_adjacent_pairs(num_summaries, args.max_pairs_per_sample)
                
            else:
                x_q = features.unsqueeze(0).to(device)
                noisy_features = add_gaussian_noise(features, args.noise_std)
                x_k = noisy_features.unsqueeze(0).to(device)
                summary_indices_q = summary_indices
                summary_indices_k = summary_indices
                
                pair_indices = get_noise_pairs(num_summaries, args.max_pairs_per_sample)
            
            if len(pair_indices) == 0:
                continue
            
            model.encoder_q.encoder._cached_mask = None
            model.encoder_k.encoder._cached_mask = None
            
            logits, labels = model(x_q, x_k, summary_indices_q, summary_indices_k, pair_indices)
            loss = criterion(logits, labels)
            
            loss_value = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss_value * len(pair_indices)
            total_pairs += len(pair_indices)
            num_batches += 1
            
            del x_q, x_k, logits, labels, loss
            
            if (num_batches) % 50 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        
        if total_pairs > 0:
            avg_loss = total_loss / total_pairs
        else:
            avg_loss = 0
        
        log_msg = f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.6f} - Pairs: {total_pairs} - LR: {scheduler.get_last_lr()[0]:.2e}"
        
        if args.positive_mode == "mix":
            total_mode_samples = mode_counts["adjacent"] + mode_counts["noise"]
            if total_mode_samples > 0:
                log_msg += f" - Mode: Adj={mode_counts['adjacent']} Noise={mode_counts['noise']}"
        
        print(log_msg)
        
        if run is not None:
            swanlab.log({
                "moco/loss": avg_loss,
                "moco/pairs": total_pairs,
                "moco/lr": scheduler.get_last_lr()[0]
            }, step=epoch)
        
        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.encoder_q.state_dict(),
                'config': config,
                'loss': avg_loss,
                'positive_mode': args.positive_mode
            }, os.path.join(args.output_dir, 'moco_summary_best.pt'))
            print(f"  -> Saved best model (loss: {avg_loss:.6f})")
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.encoder_q.state_dict(),
                'config': config,
                'loss': avg_loss,
                'positive_mode': args.positive_mode
            }, os.path.join(args.output_dir, f'moco_summary_epoch{epoch+1}.pt'))
    
    print(f"\nSummary Token Level MoCo training completed! Best loss: {best_loss:.6f}")
    
    if run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()

