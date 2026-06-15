"""
Connector Dynamic Builder

Supports dynamic loading of connector classes through module paths
"""

import importlib
import json
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path

import torch.nn as nn

from .base import BaseConnector, ConnectorConfig


REGISTERED_CONNECTORS: Dict[str, str] = {
    "mlp": "model.connectors.mlp_connector.MLPConnector",
    "qformer": "model.connectors.qformer_connector.QFormerConnector",
}


def _import_class(class_path: str) -> Type:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def build_connector(
    connector_path: str,
    mm_hidden_size: int,
    hidden_size: int,
    connector_args: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BaseConnector:
    connector_args = connector_args or {}
    
    if connector_path in REGISTERED_CONNECTORS:
        full_path = REGISTERED_CONNECTORS[connector_path]
    else:
        full_path = connector_path
    
    try:
        connector_cls = _import_class(full_path)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Failed to import connector from '{full_path}': {e}\n"
            f"Available registered connectors: {list(REGISTERED_CONNECTORS.keys())}"
        )
    
    if not issubclass(connector_cls, BaseConnector):
        raise TypeError(
            f"Connector class '{full_path}' must inherit from BaseConnector"
        )
    
    config_cls = connector_cls.get_config_class()
    
    config_dict = {
        "mm_hidden_size": mm_hidden_size,
        "hidden_size": hidden_size,
        **connector_args,
        **kwargs
    }
    
    valid_fields = set(config_cls.__dataclass_fields__.keys())
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    
    config = config_cls(**filtered_config)
    
    connector = connector_cls(config)
    
    return connector


def build_connector_from_config(config_path: str) -> BaseConnector:
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return build_connector(
        connector_path=config["connector_path"],
        mm_hidden_size=config["mm_hidden_size"],
        hidden_size=config["hidden_size"],
        connector_args=config.get("connector_args", {}),
    )


def register_connector(name: str, class_path: str):
    REGISTERED_CONNECTORS[name] = class_path


def list_registered_connectors() -> Dict[str, str]:
    return REGISTERED_CONNECTORS.copy()


def build_vision_projector_v2(config, delay_load=False, **kwargs):
    mm_hidden_size = getattr(config, "mm_hidden_size", 1024)
    hidden_size = getattr(config, "hidden_size", 4096)
    
    connector_path = getattr(config, "mm_projector_path", None)
    
    if connector_path is None:
        projector_type = getattr(config, "mm_projector_type", "mlp")
        if projector_type == "linear":
            connector_path = "mlp"
            connector_args = {"num_layers": 1}
        elif projector_type.startswith("mlp") and "gelu" in projector_type:
            import re
            match = re.match(r"mlp(\d+)x_gelu", projector_type)
            if match:
                connector_path = "mlp"
                connector_args = {"num_layers": int(match.group(1)), "activation": "gelu"}
            else:
                connector_path = "mlp"
                connector_args = {"num_layers": 2}
        elif projector_type == "identity":
            return nn.Identity()
        else:
            connector_path = "mlp"
            connector_args = {"num_layers": 2}
    else:
        connector_args_raw = getattr(config, "mm_projector_args", None)
        if connector_args_raw is None:
            connector_args = {}
        elif isinstance(connector_args_raw, str):
            connector_args = json.loads(connector_args_raw)
        else:
            connector_args = connector_args_raw
    
    return build_connector(
        connector_path=connector_path,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        connector_args=connector_args,
        **kwargs
    )
