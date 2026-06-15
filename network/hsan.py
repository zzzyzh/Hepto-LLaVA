"""
Hierarchical Sparse Attention Network

Core Components:
- HSANConfig: Model configuration
- HSANMaskGenerator: Sparse attention mask generator
- MultiheadSparseAttention: Multi-head sparse attention
- HSANBlock: Single Transformer Block
- HSANetwork: Complete network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass


@dataclass
class HSANConfig:
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dim: int = 2048
    layers: int = 6
    pack_size: int = 9
    norm_first: bool = True
    mask_type: str = "hsan"


class LongNetMaskGenerator:
    """
    LongNet-style dilated attention mask generator
    
    Dilated Attention strategy:
    - Each attention head uses different dilation rate
    - Dilation rates grow exponentially: [1, 2, 4, 8, ...]
    - Each token can see itself and tokens at dilation rate intervals
    - Global token (index 0) is visible to all tokens
    
    Reference: LongNet - Scaling Transformers to 1,000,000,000 Tokens
    """
    
    @staticmethod
    def build_mask(seq_len: int, pack_size: int, device: torch.device, num_heads: int = 8) -> torch.Tensor:
        mask = torch.full((num_heads, seq_len, seq_len), float('-inf'), device=device)
        
        dilation_rates = [2 ** i for i in range(num_heads)]
        
        for head_idx in range(num_heads):
            dilation = dilation_rates[head_idx]
            
            for i in range(seq_len):
                mask[head_idx, i, 0] = 0.0
                
                mask[head_idx, i, i] = 0.0
                
                j = i - dilation
                while j >= 0:
                    mask[head_idx, i, j] = 0.0
                    j -= dilation
                
                j = i + dilation
                while j < seq_len:
                    mask[head_idx, i, j] = 0.0
                    j += dilation
        
        return mask
    
    @staticmethod
    def build_mask_single(seq_len: int, pack_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        dilation_rates = [1, 2, 4, 8]
        
        for i in range(seq_len):
            mask[i, 0] = 0.0
            mask[i, i] = 0.0
            
            for dilation in dilation_rates:
                j = i - dilation
                while j >= 0:
                    mask[i, j] = 0.0
                    j -= dilation
                
                j = i + dilation
                while j < seq_len:
                    mask[i, j] = 0.0
                    j += dilation
        
        return mask


class HSANMaskGenerator:
    
    @staticmethod
    def build_mask(seq_len: int, pack_size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        block_stride = pack_size + 1
        

        mask[:, 0] = 0.0
        
        mask[0, 0] = 0.0
        
        rest_len = seq_len - 1
        num_blocks = math.ceil(rest_len / block_stride)
        
        summary_indices = []
        
        for k in range(num_blocks):
            start_idx = 1 + k * block_stride
            end_idx = min(start_idx + block_stride, seq_len)
            
            summ_idx = end_idx - 1
            summary_indices.append(summ_idx)
            
            patch_indices = torch.arange(start_idx, summ_idx, device=device)
            
            if len(patch_indices) > 0:

                p_grid_x, p_grid_y = torch.meshgrid(patch_indices, patch_indices, indexing='ij')
                mask[p_grid_x, p_grid_y] = 0.0
                

                mask[summ_idx, patch_indices] = 0.0
            

            mask[summ_idx, summ_idx] = 0.0


        if len(summary_indices) > 0:
            s_tensor = torch.tensor(summary_indices, device=device)
            s_grid_x, s_grid_y = torch.meshgrid(s_tensor, s_tensor, indexing='ij')
            mask[s_grid_x, s_grid_y] = 0.0
            
        return mask


class MultiheadSparseAttention(nn.Module):
    
    def __init__(self, args: HSANConfig):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads
        self.head_dim = args.embed_dim // args.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(args.embed_dim, args.embed_dim)
        self.k_proj = nn.Linear(args.embed_dim, args.embed_dim)
        self.v_proj = nn.Linear(args.embed_dim, args.embed_dim)
        self.out_proj = nn.Linear(args.embed_dim, args.embed_dim)
        
        self.dropout = nn.Dropout(args.attention_dropout)

    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (Batch, Seq_Len, Dim)
            key: (Batch, Seq_Len, Dim)
            value: (Batch, Seq_Len, Dim)
            attn_mask: (Seq_Len, Seq_Len) with 0.0 and -inf
            
        Returns:
            output: (Batch, Seq_Len, Dim)
        """
        bsz, tgt_len, _ = query.size()
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        
        if attn_mask is not None:
            if attn_mask.dtype != attn_weights.dtype:
                attn_mask = attn_mask.to(attn_weights.dtype)
            attn_weights = attn_weights + attn_mask
        
        attn_weights = torch.clamp(attn_weights, min=-1e4, max=1e4)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    
    def __init__(self, args: HSANConfig):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.ffn_dim)
        self.fc2 = nn.Linear(args.ffn_dim, args.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class HSANBlock(nn.Module):
    
    def __init__(self, args: HSANConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.attn = MultiheadSparseAttention(args)
        self.norm2 = nn.LayerNorm(args.embed_dim)
        self.ffn = FeedForward(args)
        self.norm_first = args.norm_first

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, x, x, attn_mask))
            x = self.norm2(x + self.ffn(x))
        return x


class HSANetwork(nn.Module):
    """
    Hierarchical Sparse Attention Network
    
    Input structure: [Global, P₁, P₂, ..., P₉, S₁, P₁₀, P₁₁, ..., P₁₈, S₂, ...]
    
    Where:
    - Global Token at index 0
    - Each Block contains pack_size Patches + 1 Summary
    - Summary Token at the end of each Block
    """
    
    def __init__(self, args: HSANConfig, mask_cache=None):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([HSANBlock(args) for _ in range(args.layers)])
        self.norm = nn.LayerNorm(args.embed_dim)
        
        self.input_proj = nn.Linear(512, args.embed_dim) if 512 != args.embed_dim else nn.Identity()
        
        self._cached_mask = None
        self._cached_seq_len = None
        
        self.mask_type = args.mask_type if hasattr(args, 'mask_type') else "hsan"
        
        self.mask_cache = mask_cache
        if self.mask_type == "longnet" and mask_cache is not None:
            print(f"Using mask type: {self.mask_type} (with precomputed cache)")
        else:
            print(f"Using mask type: {self.mask_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        seq_len = x.size(1)
        
        if self._cached_mask is None or self._cached_seq_len != seq_len:
            if self.mask_type == "longnet":
                if self.mask_cache is not None:
                    try:
                        self._cached_mask = self.mask_cache.load_mask(
                            seq_len, self.args.num_heads, x.device
                        )
                    except FileNotFoundError as e:
                        print(f"Warning: {e}")
                        print("Falling back to dynamic mask generation")
                        self._cached_mask = LongNetMaskGenerator.build_mask_single(
                            seq_len, self.args.pack_size, x.device
                        )
                else:
                    self._cached_mask = LongNetMaskGenerator.build_mask_single(
                        seq_len, self.args.pack_size, x.device
                    )
            else:
                self._cached_mask = HSANMaskGenerator.build_mask(
                    seq_len, self.args.pack_size, x.device
                )
            self._cached_seq_len = seq_len
        
        mask = self._cached_mask
        if mask.device != x.device:
            mask = mask.to(x.device)
            self._cached_mask = mask
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return x

