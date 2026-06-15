"""
HSAN-MAE: Masked Autoencoder Model

Components:
- SequenceParser: Sequence structure parser
- DynamicMaskGenerator: Dynamic attention mask generator
- MAEDecoder: MAE decoder
- HSANForMAE: Complete MAE model
"""

import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.checkpoint import checkpoint
from typing import Dict, Tuple, Optional, List

from network import HSANetwork, HSANConfig, HSANMaskGenerator


def get_2d_sinusoidal_pos_embed_for_seq(
    pack_coords: List[Tuple[float, float]], 
    seq_len: int, 
    embed_dim: int, 
    pack_size: int = 9,
    temperature: float = 10000.0
) -> torch.Tensor:
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    
    pos_embed = torch.zeros(seq_len, embed_dim)
    
    if len(pack_coords) == 0:
        return pos_embed.unsqueeze(0)
    
    coords = np.array(pack_coords, dtype=np.float32)
    num_packs = coords.shape[0]
    
    if num_packs > 0:
        max_coord = coords.max()
        if max_coord > 0:
            coords = coords / max_coord * (2 * np.pi)
    
    num_freqs = embed_dim // 4
    omega = np.arange(num_freqs, dtype=np.float32)
    omega = 1.0 / (temperature ** (omega / num_freqs))
    
    x_coords = coords[:, 0:1]
    x_embed = x_coords * omega[None, :]
    x_pos = np.concatenate([np.sin(x_embed), np.cos(x_embed)], axis=1)
    
    y_coords = coords[:, 1:2]
    y_embed = y_coords * omega[None, :]
    y_pos = np.concatenate([np.sin(y_embed), np.cos(y_embed)], axis=1)
    
    pack_pos_embed = np.concatenate([x_pos, y_pos], axis=1)
    pack_pos_embed = torch.from_numpy(pack_pos_embed).float()
    
    tokens_per_pack = pack_size + 1
    
    for pack_idx in range(num_packs):
        pack_start = 1 + pack_idx * tokens_per_pack
        for token_offset in range(tokens_per_pack):
            token_idx = pack_start + token_offset
            if token_idx < seq_len:
                pos_embed[token_idx] = pack_pos_embed[pack_idx]
    
    return pos_embed.unsqueeze(0)


class SequenceParser:
    
    TOKEN_GLOBAL = 0
    TOKEN_PATCH = 1
    TOKEN_SUMMARY = 2
    
    def __init__(self, pack_size: int):
        self.pack_size = pack_size
        self.block_stride = pack_size + 1
    
    def parse(self, seq_len: int) -> Dict:
        global_idx = 0
        patch_indices = []
        summary_indices = []
        pack_info = []
        
        rest_len = seq_len - 1
        num_blocks = math.ceil(rest_len / self.block_stride)
        
        for k in range(num_blocks):
            start_idx = 1 + k * self.block_stride
            end_idx = min(start_idx + self.block_stride, seq_len)
            
            summ_idx = end_idx - 1
            summary_indices.append(summ_idx)
            
            pack_patch_indices = list(range(start_idx, summ_idx))
            patch_indices.extend(pack_patch_indices)
            
            pack_info.append({
                'pack_idx': k,
                'start': start_idx,
                'end': summ_idx,
                'summary_idx': summ_idx,
                'patch_indices': pack_patch_indices
            })
        
        return {
            'global_idx': global_idx,
            'patch_indices': patch_indices,
            'summary_indices': summary_indices,
            'pack_info': pack_info,
            'num_packs': num_blocks
        }
    
    def get_token_info(self, original_idx: int, seq_len: int) -> Dict:
        if original_idx == 0:
            return {'token_type': self.TOKEN_GLOBAL, 'pack_idx': -1}
        
        pos_in_rest = original_idx - 1
        pack_idx = pos_in_rest // self.block_stride
        pos_in_pack = pos_in_rest % self.block_stride
        
        if pos_in_pack == self.pack_size:
            return {'token_type': self.TOKEN_SUMMARY, 'pack_idx': pack_idx}
        else:
            return {'token_type': self.TOKEN_PATCH, 'pack_idx': pack_idx}


class DynamicMaskGenerator:
    
    @staticmethod
    def build_mask(
        visible_indices: torch.Tensor,
        original_seq_len: int,
        pack_size: int,
        device: torch.device
    ) -> torch.Tensor:
        if visible_indices.dim() == 2:
            visible_indices = visible_indices[0]
        
        visible_indices = visible_indices.cpu().tolist()
        num_visible = len(visible_indices)
        
        mask = torch.full((num_visible, num_visible), float('-inf'), device=device)
        
        parser = SequenceParser(pack_size)
        
        token_infos = []
        for orig_idx in visible_indices:
            info = parser.get_token_info(orig_idx, original_seq_len)
            token_infos.append(info)
        
        global_pos = None
        patch_positions_by_pack = {}
        summary_positions = []
        summary_pack_map = {}
        
        for pos, info in enumerate(token_infos):
            if info['token_type'] == SequenceParser.TOKEN_GLOBAL:
                global_pos = pos
            elif info['token_type'] == SequenceParser.TOKEN_PATCH:
                pack_idx = info['pack_idx']
                if pack_idx not in patch_positions_by_pack:
                    patch_positions_by_pack[pack_idx] = []
                patch_positions_by_pack[pack_idx].append(pos)
            elif info['token_type'] == SequenceParser.TOKEN_SUMMARY:
                summary_positions.append(pos)
                summary_pack_map[pos] = info['pack_idx']
        
        if global_pos is not None:
            mask[:, global_pos] = 0.0
            mask[global_pos, :] = float('-inf')
            mask[global_pos, global_pos] = 0.0
        
        for pack_idx, positions in patch_positions_by_pack.items():
            for i in positions:
                for j in positions:
                    mask[i, j] = 0.0
        
        for summ_pos in summary_positions:
            pack_idx = summary_pack_map[summ_pos]
            mask[summ_pos, summ_pos] = 0.0
            if pack_idx in patch_positions_by_pack:
                for patch_pos in patch_positions_by_pack[pack_idx]:
                    mask[summ_pos, patch_pos] = 0.0
        
        for i in summary_positions:
            for j in summary_positions:
                mask[i, j] = 0.0
        
        return mask


class MAEDecoder(nn.Module):
    
    def __init__(
        self, 
        embed_dim: int = 512, 
        decoder_embed_dim: int = 256,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        patch_dim: int = 512,
        max_seq_len: int = 15000,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True,
        use_fixed_pos_embed: bool = False,
        pack_size: int = 9
    ):
        super().__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.max_seq_len = max_seq_len
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fixed_pos_embed = use_fixed_pos_embed
        self.pack_size = pack_size
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim) * 0.02)
        
        if not use_fixed_pos_embed:
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, max_seq_len, decoder_embed_dim) * 0.02)
        else:
            self.decoder_pos_embed = None
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=decoder_num_heads,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)
        
    def forward(
        self, 
        x_visible: torch.Tensor,
        visible_indices: torch.Tensor,
        mask_indices: torch.Tensor,
        seq_len: int,
        pack_coords: List[Tuple[float, float]] = None
    ) -> torch.Tensor:
        bsz = x_visible.size(0)
        num_visible = x_visible.size(1)
        
        if visible_indices.dim() == 1:
            visible_indices = visible_indices.unsqueeze(0).expand(bsz, -1)
        if mask_indices.dim() == 1:
            mask_indices = mask_indices.unsqueeze(0).expand(bsz, -1)
        
        num_masked = mask_indices.size(1)
        
        x = self.decoder_embed(x_visible)
        mask_tokens = self.mask_token.expand(bsz, num_masked, -1)
        
        x_full = torch.zeros(bsz, seq_len, self.decoder_embed_dim, device=x.device, dtype=x.dtype)
        
        for b in range(bsz):
            x_full[b, visible_indices[b]] = x[b]
            x_full[b, mask_indices[b]] = mask_tokens[b]
        
        if self.use_fixed_pos_embed:
            if pack_coords is None:
                pack_coords = []
            pos_embed = get_2d_sinusoidal_pos_embed_for_seq(
                pack_coords, seq_len, self.decoder_embed_dim, self.pack_size
            ).to(x.device)
        else:
            if seq_len <= self.max_seq_len:
                pos_embed = self.decoder_pos_embed[:, :seq_len, :]
            else:
                pos_embed_2d = self.decoder_pos_embed.transpose(1, 2)
                pos_embed_2d = torch.nn.functional.interpolate(
                    pos_embed_2d, size=seq_len, mode='linear', align_corners=False
                )
                pos_embed = pos_embed_2d.transpose(1, 2)
        
        x_full = x_full + pos_embed
        
        if self.use_gradient_checkpointing and self.training:
            def decoder_forward(x):
                return self.decoder_blocks(x)
            x_full = checkpoint(decoder_forward, x_full, use_reentrant=False)
        else:
            x_full = self.decoder_blocks(x_full)
        
        x_full = self.decoder_norm(x_full)
        pred = self.decoder_pred(x_full)
        
        masked_pred = torch.zeros(bsz, num_masked, pred.size(-1), device=pred.device, dtype=pred.dtype)
        for b in range(bsz):
            masked_pred[b] = pred[b, mask_indices[b]]
        
        return masked_pred


class HSANForMAE(nn.Module):
    
    def __init__(
        self, 
        config: HSANConfig, 
        mask_ratio: float = 0.75, 
        use_gradient_checkpointing: bool = True,
        use_fixed_pos_embed: bool = False,
        mask_cache = None
    ):
        super().__init__()
        self.config = config
        self.mask_ratio = mask_ratio
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_fixed_pos_embed = use_fixed_pos_embed
        
        self.seq_parser = SequenceParser(config.pack_size)
        self.encoder = HSANetwork(config, mask_cache=mask_cache)
        
        self.decoder = MAEDecoder(
            embed_dim=config.embed_dim,
            decoder_embed_dim=config.embed_dim // 2,
            decoder_depth=2,
            decoder_num_heads=4,
            patch_dim=512,
            dropout=config.dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_fixed_pos_embed=use_fixed_pos_embed,
            pack_size=config.pack_size
        )
    
    def patch_level_masking(self, x: torch.Tensor, mask_ratio: float = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        bsz, seq_len, dim = x.size()
        device = x.device
        
        parsed = self.seq_parser.parse(seq_len)
        global_idx = parsed['global_idx']
        patch_indices = torch.tensor(parsed['patch_indices'], device=device, dtype=torch.long)
        summary_indices = torch.tensor(parsed['summary_indices'], device=device, dtype=torch.long)
        
        num_patches = len(patch_indices)
        num_keep_patches = max(1, int(num_patches * (1 - mask_ratio)))
        
        noise = torch.rand(bsz, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        keep_patch_local = ids_shuffle[:, :num_keep_patches]
        mask_patch_local = ids_shuffle[:, num_keep_patches:]
        
        keep_patch_global = torch.gather(
            patch_indices.unsqueeze(0).expand(bsz, -1), dim=1, index=keep_patch_local
        )
        mask_patch_global = torch.gather(
            patch_indices.unsqueeze(0).expand(bsz, -1), dim=1, index=mask_patch_local
        )
        
        visible_list = []
        for b in range(bsz):
            vis = torch.cat([
                torch.tensor([global_idx], device=device, dtype=torch.long),
                keep_patch_global[b],
                summary_indices
            ])
            vis_sorted, _ = torch.sort(vis)
            visible_list.append(vis_sorted)
        
        visible_indices = torch.stack(visible_list)
        
        num_visible = visible_indices.size(1)
        x_visible = torch.zeros(bsz, num_visible, dim, device=device, dtype=x.dtype)
        for b in range(bsz):
            x_visible[b] = x[b, visible_indices[b]]
        
        mask = torch.zeros(bsz, seq_len, device=device)
        for b in range(bsz):
            mask[b, mask_patch_global[b]] = 1.0
        
        return x_visible, mask, visible_indices, mask_patch_global, mask_patch_global
    
    def pack_level_masking(self, x: torch.Tensor, mask_ratio: float = None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        bsz, seq_len, dim = x.size()
        device = x.device
        
        parsed = self.seq_parser.parse(seq_len)
        global_idx = parsed['global_idx']
        pack_info = parsed['pack_info']
        num_packs = parsed['num_packs']
        
        num_keep_packs = max(1, int(num_packs * (1 - mask_ratio)))
        
        noise = torch.rand(bsz, num_packs, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        keep_pack_ids = ids_shuffle[:, :num_keep_packs]
        
        visible_list = []
        mask_indices_list = []
        target_indices_list = []
        
        for b in range(bsz):
            visible_tokens = [global_idx]
            masked_tokens = []
            target_tokens = []
            
            keep_packs = keep_pack_ids[b].tolist()
            
            for pack in pack_info:
                pack_idx = pack['pack_idx']
                if pack_idx in keep_packs:
                    visible_tokens.extend(pack['patch_indices'])
                    visible_tokens.append(pack['summary_idx'])
                else:
                    masked_tokens.extend(pack['patch_indices'])
                    masked_tokens.append(pack['summary_idx'])
                    target_tokens.extend(pack['patch_indices'])
                    target_tokens.append(pack['summary_idx'])
            
            visible_tokens = sorted(visible_tokens)
            visible_list.append(torch.tensor(visible_tokens, device=device, dtype=torch.long))
            mask_indices_list.append(torch.tensor(masked_tokens, device=device, dtype=torch.long))
            target_indices_list.append(torch.tensor(target_tokens, device=device, dtype=torch.long))
        
        max_visible = max(v.size(0) for v in visible_list)
        max_masked = max(m.size(0) for m in mask_indices_list)
        max_target = max(t.size(0) for t in target_indices_list)
        
        visible_indices = torch.zeros(bsz, max_visible, dtype=torch.long, device=device)
        mask_indices = torch.zeros(bsz, max_masked, dtype=torch.long, device=device)
        target_indices = torch.zeros(bsz, max_target, dtype=torch.long, device=device)
        
        for b in range(bsz):
            visible_indices[b, :visible_list[b].size(0)] = visible_list[b]
            mask_indices[b, :mask_indices_list[b].size(0)] = mask_indices_list[b]
            target_indices[b, :target_indices_list[b].size(0)] = target_indices_list[b]
        
        x_visible = torch.zeros(bsz, max_visible, dim, device=device, dtype=x.dtype)
        for b in range(bsz):
            actual_visible = visible_list[b].size(0)
            x_visible[b, :actual_visible] = x[b, visible_list[b]]
        
        mask = torch.zeros(bsz, seq_len, device=device)
        for b in range(bsz):
            mask[b, mask_indices_list[b]] = 1.0
        
        return x_visible, mask, visible_indices, mask_indices, target_indices
    
    def forward_encoder(
        self, 
        x_visible: torch.Tensor, 
        visible_indices: torch.Tensor,
        original_seq_len: int = None,
        use_dynamic_mask: bool = False
    ) -> torch.Tensor:
        bsz, num_visible, _ = x_visible.size()
        
        x = self.encoder.input_proj(x_visible)
        
        if use_dynamic_mask and original_seq_len is not None:
            mask = DynamicMaskGenerator.build_mask(
                visible_indices, original_seq_len, self.config.pack_size, x.device
            )
        else:
            mask = HSANMaskGenerator.build_mask(num_visible, self.config.pack_size, x.device)
        
        if self.use_gradient_checkpointing and self.training:
            for layer in self.encoder.layers:
                def layer_forward(x_in, mask_in, layer_module):
                    return layer_module(x_in, mask_in)
                x = checkpoint(layer_forward, x, mask, layer, use_reentrant=False)
        else:
            for layer in self.encoder.layers:
                x = layer(x, mask)
        
        x = self.encoder.norm(x)
        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: float = None,
        masking_mode: str = 'patch',
        pack_coords: List[Tuple[float, float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        
        if masking_mode == 'patch':
            x_visible, mask, visible_indices, mask_indices, target_indices = \
                self.patch_level_masking(x, mask_ratio)
            use_dynamic_mask = True
        else:
            x_visible, mask, visible_indices, mask_indices, target_indices = \
                self.pack_level_masking(x, mask_ratio)
            use_dynamic_mask = False
        
        encoded = self.forward_encoder(
            x_visible, visible_indices, original_seq_len=seq_len, use_dynamic_mask=use_dynamic_mask
        )
        
        pred = self.decoder(encoded, visible_indices, target_indices, seq_len, pack_coords=pack_coords)
        
        bsz = x.size(0)
        num_target = target_indices.size(1)
        target = torch.zeros(bsz, num_target, x.size(-1), device=x.device, dtype=x.dtype)
        for b in range(bsz):
            target[b] = x[b, target_indices[b]]
        
        loss = ((pred - target) ** 2).mean()
        
        return loss, pred, mask
    
    def get_representations(
        self, 
        x: torch.Tensor, 
        text_scaler=None, 
        target_summary_dim=None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        
        encoded = self.encoder(x)
        
        parsed = self.seq_parser.parse(seq_len)
        summary_indices = parsed['summary_indices']
        
        global_token = encoded[:, 0, :]
        
        summary_indices_tensor = torch.tensor(summary_indices, device=encoded.device)
        summary_tokens = encoded[:, summary_indices_tensor, :]
        
        if text_scaler is not None:
            original_summaries = x[:, summary_indices_tensor, :]
            clip_scores = text_scaler.compute_scale(original_summaries)
            weights = torch.softmax(clip_scores / 0.05, dim=1)
            weights_expanded = weights.unsqueeze(-1)
            summary_pooled = (summary_tokens * weights_expanded).sum(dim=1)
        else:
            summary_pooled = summary_tokens.mean(dim=1)
        
        representation = torch.cat([global_token, summary_pooled], dim=-1)
        
        return representation
    
    def get_global_pooled_representation(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        global_pooled = encoded.mean(dim=1)
        return global_pooled

