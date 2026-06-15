"""
Base Connector

Defines the base interface for vision-language connectors
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ConnectorConfig:
    mm_hidden_size: int
    hidden_size: int
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConnectorConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class BaseConnector(nn.Module, ABC):
    
    def __init__(self, config: ConnectorConfig):
        super().__init__()
        self.config = config
        self.mm_hidden_size = config.mm_hidden_size
        self.hidden_size = config.hidden_size
    
    @abstractmethod
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        pass
    
    @classmethod
    def get_config_class(cls):
        return ConnectorConfig
    
    def get_output_token_count(self, input_token_count: int) -> int:
        return input_token_count
