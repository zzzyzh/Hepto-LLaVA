"""
HSAN-MAE Self-Supervised Pretraining

Curriculum Learning Strategy:
- Phase 1.1: Patch-level masking - Mask only patch tokens, keep Global and Summary
- Phase 1.2: Pack-level masking - Mask by pack unit, keep Global token
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

# SwanLab for experiment tracking
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not installed. Install with: pip install swanlab")

from network import HSANConfig
from model import HSANForMAE
from utils import (
    set_seed,
    MAEPretrainDataset,
    collate_fn_mae,
    CurriculumScheduler,
    LinearWarmupCosineAnnealingLR,
    pretrain_epoch
)


def main():
    parser = argparse.ArgumentParser(description="HSAN-MAE Self-Supervised Pretraining")
    
    parser.add_argument("--feature-dir", type=str, required=True, 
                        help="Pretrain feature directory")
    parser.add_argument("--output-dir", type=str, default="./outputs_mae", help="Output directory")
    
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Mask ratio")
    parser.add_argument("--epochs", type=int, default=100, help="Pretrain epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Pretrain learning rate")
    parser.add_argument("--curriculum-transition", type=float, default=0.5, 
                        help="Curriculum learning phase transition ratio")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Warmup epochs")
    parser.add_argument("--warmup-lr", type=float, default=1e-6,
                        help="Warmup initial learning rate")
    
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--pack-size", type=int, default=9, help="Pack size")
    parser.add_argument("--use-fixed-pos-embed", action="store_true",
                        help="Use fixed 2D sinusoidal position embedding")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", 
                        action="store_false", help="Disable gradient checkpointing")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run = None
    if SWANLAB_AVAILABLE:
        run = swanlab.init(
            project="HSAN-MAE-Pretrain",
            experiment_name=os.path.basename(args.output_dir),
            description="HSAN-MAE Self-Supervised Pretraining",
            config=vars(args),
            logdir=os.path.join(args.output_dir, "swanlab_logs")
        )
        print(f"SwanLab initialized. Logs will be saved to {os.path.join(args.output_dir, 'swanlab_logs')}")
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = HSANConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        layers=args.layers,
        pack_size=args.pack_size
    )
    
    print("\n" + "=" * 60)
    print("HSAN-MAE Pretraining with Curriculum Learning")
    print("=" * 60)
    
    mae_model = HSANForMAE(
        config, 
        mask_ratio=args.mask_ratio,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_fixed_pos_embed=args.use_fixed_pos_embed
    ).to(device)
    
    total_params = sum(p.numel() for p in mae_model.parameters())
    trainable_params = sum(p.numel() for p in mae_model.parameters() if p.requires_grad)
    print(f"MAE Model Parameters: {total_params:,} (trainable: {trainable_params:,})")
    print(f"Gradient Checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
    print(f"Position Embedding: {'Fixed 2D sinusoidal' if args.use_fixed_pos_embed else 'Learnable'}")
    
    pretrain_dataset = MAEPretrainDataset(
        args.feature_dir, 
        return_pack_coords=args.use_fixed_pos_embed
    )
    pretrain_loader = DataLoader(
        pretrain_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=collate_fn_mae
    )
    
    curriculum = CurriculumScheduler(
        args.epochs, 
        transition_ratio=args.curriculum_transition
    )
    
    optimizer = optim.AdamW(mae_model.parameters(), lr=args.lr)
    
    if args.warmup_epochs > 0:
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            warmup_lr=args.warmup_lr,
            eta_min=1e-6
        )
        print(f"Using LinearWarmupCosineAnnealingLR: {args.warmup_epochs} warmup epochs")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        print(f"Using CosineAnnealingLR (no warmup)")
    
    best_loss = float('inf')
    best_loss_phase1 = float('inf')
    best_loss_phase2 = float('inf')
    
    print(f"\nStarting MAE pretraining for {args.epochs} epochs...")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Curriculum transition at epoch: {curriculum.transition_epoch}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    for epoch in range(args.epochs):
        masking_mode = curriculum.get_masking_mode(epoch)
        phase_name = curriculum.get_phase_name(epoch)
        
        loss = pretrain_epoch(
            mae_model, pretrain_loader, optimizer, device, 
            args.mask_ratio, masking_mode=masking_mode,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} [{phase_name}] - "
              f"Loss: {loss:.6f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if run is not None:
            swanlab.log({
                "pretrain/loss": loss,
                "pretrain/learning_rate": scheduler.get_last_lr()[0],
                "pretrain/phase": 1 if masking_mode == 'patch' else 2,
                "pretrain/epoch": epoch + 1
            }, step=epoch)
        
        if loss < best_loss:
            best_loss = loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mae_model.state_dict(),
                'config': config,
                'mask_ratio': args.mask_ratio,
                'masking_mode': masking_mode,
                'loss': loss,
                'use_fixed_pos_embed': args.use_fixed_pos_embed
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'mae_pretrained_best.pt'))
            print(f"  -> Saved best model (loss: {loss:.6f})")
        
        if masking_mode == 'patch' and loss < best_loss_phase1:
            best_loss_phase1 = loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mae_model.state_dict(),
                'config': config,
                'mask_ratio': args.mask_ratio,
                'masking_mode': masking_mode,
                'loss': loss,
                'use_fixed_pos_embed': args.use_fixed_pos_embed
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'mae_pretrained_phase1_best.pt'))
        elif masking_mode == 'pack' and loss < best_loss_phase2:
            best_loss_phase2 = loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mae_model.state_dict(),
                'config': config,
                'mask_ratio': args.mask_ratio,
                'masking_mode': masking_mode,
                'loss': loss,
                'use_fixed_pos_embed': args.use_fixed_pos_embed
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'mae_pretrained_phase2_best.pt'))
        
        if epoch == curriculum.transition_epoch - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mae_model.state_dict(),
                'config': config,
                'mask_ratio': args.mask_ratio,
                'masking_mode': masking_mode,
                'loss': loss,
                'use_fixed_pos_embed': args.use_fixed_pos_embed
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'mae_pretrained_phase1_final.pt'))
            print(f"  -> Saved Phase 1 final checkpoint")
        
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': mae_model.state_dict(),
                'config': config,
                'mask_ratio': args.mask_ratio,
                'masking_mode': masking_mode,
                'loss': loss,
                'use_fixed_pos_embed': args.use_fixed_pos_embed
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'mae_pretrained_epoch{epoch+1}.pt'))
    
    print(f"\nMAE pretraining completed!")
    print(f"  Best overall loss: {best_loss:.6f}")
    print(f"  Best Phase 1 (patch-level) loss: {best_loss_phase1:.6f}")
    print(f"  Best Phase 2 (pack-level) loss: {best_loss_phase2:.6f}")
    
    if run is not None:
        swanlab.finish()
        print(f"SwanLab logs saved to {os.path.join(args.output_dir, 'swanlab_logs')}")


if __name__ == "__main__":
    main()

