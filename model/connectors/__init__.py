"""
Vision-Language Connector Module

Provides multiple connector implementations for mapping visual features to LLM hidden space:
- MLPConnector: Simple MLP-based projection
- QFormerConnector: Q-Former-based cross-attention aggregation

Usage:
1. Build with pre-registered name:
   connector = build_connector("mlp", mm_hidden_size=1024, hidden_size=4096)

2. Build with full module path:
   connector = build_connector(
       "model.connectors.mlp_connector.MLPConnector",
       mm_hidden_size=1024,
       hidden_size=4096
   )

3. Build from config file:
   connector = build_connector_from_config("config.json")
"""

from .base import BaseConnector, ConnectorConfig
from .mlp_connector import MLPConnector, MLPConnectorConfig, build_mlp_connector
from .qformer_connector import QFormerConnector, QFormerConnectorConfig, build_qformer_connector
from .builder import (
    build_connector,
    build_connector_from_config,
    build_vision_projector_v2,
    register_connector,
    list_registered_connectors,
    REGISTERED_CONNECTORS,
)

__all__ = [
    "BaseConnector",
    "ConnectorConfig",
    "MLPConnector",
    "MLPConnectorConfig",
    "build_mlp_connector",
    "QFormerConnector",
    "QFormerConnectorConfig", 
    "build_qformer_connector",
    "build_connector",
    "build_connector_from_config",
    "build_vision_projector_v2",
    "register_connector",
    "list_registered_connectors",
    "REGISTERED_CONNECTORS",
]
