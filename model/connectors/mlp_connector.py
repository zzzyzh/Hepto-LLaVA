"""
MLP Connector

Multi-layer perceptron connector for mapping visual features to LLM hidden space
Supports configurable layers and activation functions
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from .base import BaseConnector, ConnectorConfig


@dataclass
class MLPConnectorConfig(ConnectorConfig):
    num_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.0
    use_layer_norm: bool = False
    intermediate_size: Optional[int] = None


class MLPConnector(BaseConnector):
    
    def __init__(self, config: MLPConnectorConfig):
        super().__init__(config)
        
        self.num_layers = config.num_layers
        self.activation_name = config.activation
        self.use_layer_norm = config.use_layer_norm
        self.intermediate_size = config.intermediate_size or config.hidden_size
        
        layers = []
        in_features = config.mm_hidden_size
        
        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                out_features = config.hidden_size
            else:
                out_features = self.intermediate_size
            
            layers.append(nn.Linear(in_features, out_features))
            
            layers.append(self._get_activation(config.activation))
            
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_features))
            
            in_features = out_features
        
        self.mlp = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        activation_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        if activation.lower() not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Supported: {list(activation_map.keys())}")
        return activation_map[activation.lower()]
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        input_shape = visual_features.shape
        if len(input_shape) == 2:
            visual_features = visual_features.unsqueeze(0)
        
        output = self.mlp(visual_features)
        
        if len(input_shape) == 2:
            output = output.squeeze(0)
        
        return output
    
    @classmethod
    def get_config_class(cls):
        return MLPConnectorConfig


def build_mlp_connector(
    mm_hidden_size: int,
    hidden_size: int,
    num_layers: int = 2,
    activation: str = "gelu",
    dropout: float = 0.0,
    use_layer_norm: bool = False,
    **kwargs
) -> MLPConnector:
    config = MLPConnectorConfig(
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation=activation,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    )
    return MLPConnector(config)
