"""
Q-Former Style Connector

BLIP-2 inspired Q-Former style connector
Uses learnable query vectors to extract information from visual features via cross-attention
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .base import BaseConnector, ConnectorConfig


@dataclass
class QFormerConnectorConfig(ConnectorConfig):
    num_query_tokens: int = 32
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    cross_attention_freq: int = 1
    use_self_attention: bool = True
    layer_norm_eps: float = 1e-6


class QFormerBlock(nn.Module):
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mm_hidden_size: int,
        dropout: float = 0.1,
        use_self_attention: bool = True,
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_self_attention = use_self_attention
        
        if use_self_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.self_attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            self.self_attn_dropout = nn.Dropout(dropout)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=mm_hidden_size,
            vdim=mm_hidden_size
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        query_tokens: torch.Tensor,
        visual_features: torch.Tensor,
        do_cross_attention: bool = True
    ) -> torch.Tensor:
        if self.use_self_attention:
            residual = query_tokens
            query_tokens = self.self_attn_norm(query_tokens)
            query_tokens, _ = self.self_attn(
                query=query_tokens,
                key=query_tokens,
                value=query_tokens
            )
            query_tokens = residual + self.self_attn_dropout(query_tokens)
        
        if do_cross_attention:
            residual = query_tokens
            query_tokens = self.cross_attn_norm(query_tokens)
            query_tokens, _ = self.cross_attn(
                query=query_tokens,
                key=visual_features,
                value=visual_features
            )
            query_tokens = residual + self.cross_attn_dropout(query_tokens)
        
        residual = query_tokens
        query_tokens = self.ffn_norm(query_tokens)
        query_tokens = residual + self.ffn(query_tokens)
        
        return query_tokens


class QFormerConnector(BaseConnector):
    
    def __init__(self, config: QFormerConnectorConfig):
        super().__init__(config)
        
        self.num_query_tokens = config.num_query_tokens
        self.cross_attention_freq = config.cross_attention_freq
        
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
        self.layers = nn.ModuleList([
            QFormerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                mm_hidden_size=config.mm_hidden_size,
                dropout=config.dropout,
                use_self_attention=config.use_self_attention,
                layer_norm_eps=config.layer_norm_eps
            )
            for _ in range(config.num_layers)
        ])
        
        self.visual_proj = None
        if config.mm_hidden_size != config.hidden_size:
            pass
        
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        input_shape = visual_features.shape
        if len(input_shape) == 2:
            visual_features = visual_features.unsqueeze(0)
        
        batch_size = visual_features.shape[0]
        
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        for i, layer in enumerate(self.layers):
            do_cross_attention = (i % self.cross_attention_freq == 0)
            query_tokens = layer(
                query_tokens=query_tokens,
                visual_features=visual_features,
                do_cross_attention=do_cross_attention
            )
        
        output = self.output_norm(query_tokens)
        
        if len(input_shape) == 2:
            output = output.squeeze(0)
        
        return output
    
    def get_output_token_count(self, input_token_count: int) -> int:
        return self.num_query_tokens
    
    @classmethod
    def get_config_class(cls):
        return QFormerConnectorConfig


def build_qformer_connector(
    mm_hidden_size: int,
    hidden_size: int,
    num_query_tokens: int = 32,
    num_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    cross_attention_freq: int = 1,
    use_self_attention: bool = True,
    **kwargs
) -> QFormerConnector:
    config = QFormerConnectorConfig(
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        num_query_tokens=num_query_tokens,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        cross_attention_freq=cross_attention_freq,
        use_self_attention=use_self_attention,
    )
    return QFormerConnector(config)
