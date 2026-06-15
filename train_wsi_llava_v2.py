#!/usr/bin/env python3
"""
WSI-LLaVA Training Script V2 (Full Integration with Custom Connector)

This script provides complete WSI-LLaVA training functionality with support
for custom vision-language connectors. By patching modules before import,
ensures custom connectors are properly used.

Key Features:
1. Support for MLP Connector and Q-Former Connector
2. Support for dynamically loading custom Connectors via module path
3. Full compatibility with original WSI-LLaVA

Usage Examples:
    # Using MLP Connector
    deepspeed train_wsi_llava_v2.py \
        --model_name_or_path /path/to/llava-v1.5-7b \
        --vision_tower /path/to/clip-vit-large-patch14-336 \
        --mm_projector_path mlp \
        --mm_projector_args '{"num_layers": 2, "activation": "gelu"}' \
        --data_path /path/to/train.json \
        --image_folder /path/to/features \
        --output_dir /path/to/output \
        --lora_enable True \
        --bf16 True
    
    # Using Q-Former Connector
    deepspeed train_wsi_llava_v2.py \
        --mm_projector_path qformer \
        --mm_projector_args '{"num_query_tokens": 32, "num_layers": 2}' \
        ...
"""

import os
import sys
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Add project path (must be before importing llava)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "third_party"))  # enables: import llava, import conch

# Import custom connector
from model.connectors import build_connector


def parse_connector_args(args_str: Optional[str]) -> Dict[str, Any]:
    """
    Parse Connector arguments JSON string, compatible with standard JSON
    and shell-corrupted non-standard format
    """
    if args_str is None or args_str == "":
        return {}
    # Handle possible quote issues
    args_str = args_str.strip()
    if args_str.startswith("'") and args_str.endswith("'"):
        args_str = args_str[1:-1]
    
    # Try standard JSON parsing
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        pass
    
    # Fallback: handle non-standard format with quotes eaten by shell
    # Example: {num_layers: 2, activation: gelu, dropout: 0.1, use_layer_norm: true}
    try:
        # Add quotes to unquoted keys: word: -> "word":
        fixed = re.sub(r'(\w+)\s*:', r'"\1":', args_str)
        # Add quotes to unquoted string values (excluding numbers, true/false/null)
        # Match ": followed by word that's not number/quote/true/false/null
        fixed = re.sub(
            r':\s*(?![\d"{\[\-]|true|false|null)([a-zA-Z_]\w*)',
            r': "\1"',
            fixed
        )
        result = json.loads(fixed)
        logger.info(f"[parse_connector_args] Parsed non-standard JSON: {result}")
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Failed to parse connector args '{args_str}': {e}")
        return {}


def build_custom_projector(config, delay_load=False, **kwargs):
    """
    Build custom Connector or use pretrained Projector
    
    This function replaces the original build_vision_projector
    Signature is fully compatible with original function
    
    Args:
        config: Model configuration object
        delay_load: Delay load flag (compatibility parameter)
        **kwargs: Other parameters
    
    Returns:
        Connector/Projector module
    """
    # Check if using pretrained projector
    use_pretrained = getattr(config, "use_pretrained_projector", False)
    
    if use_pretrained:
        # ===== Use original WSI-LLaVA projector =====
        logger.info("=" * 60)
        logger.info("[Pretrained Projector Mode] Using original WSI-LLaVA projector")
        logger.info("=" * 60)
        
        # Use original dimension config (no padding)
        mm_hidden_size = getattr(config, "mm_hidden_size", 1024)
        hidden_size = getattr(config, "hidden_size", 4096)
        projector_type = getattr(config, "mm_projector_type", "linear")
        
        logger.info(f"[Pretrained Projector] Type: {projector_type}")
        logger.info(f"[Pretrained Projector] mm_hidden_size={mm_hidden_size}, hidden_size={hidden_size}")
        
        # Build original projector (identical to WSI-LLaVA)
        if projector_type == "linear":
            projector = nn.Linear(mm_hidden_size, hidden_size)
        elif projector_type == "identity":
            class IdentityMap(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, x, *args, **kwargs):
                    return x
                @property
                def config(self):
                    return {"mm_projector_type": 'identity'}
            projector = IdentityMap()
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(mm_hidden_size, hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(hidden_size, hidden_size))
                projector = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {projector_type}")
        
        logger.info(f"[Pretrained Projector] Built successfully")
        logger.info("=" * 60)
        return projector
    
    # ===== Use custom Connector system =====
    mm_hidden_size = getattr(config, "mm_hidden_size", 1024)
    hidden_size = getattr(config, "hidden_size", 4096)
    
    # Get connector configuration
    connector_path = getattr(config, "mm_projector_path", None)
    connector_args_str = getattr(config, "mm_projector_args", None)
    
    if connector_path is not None and connector_path not in ["", "None", "none"]:
        # Use new connector system
        connector_args = parse_connector_args(connector_args_str)
        
        logger.info(f"[Custom Connector] Building: {connector_path}")
        logger.info(f"[Custom Connector] Args: {connector_args}")
        logger.info(f"[Custom Connector] mm_hidden_size={mm_hidden_size}, hidden_size={hidden_size}")
        
        connector = build_connector(
            connector_path=connector_path,
            mm_hidden_size=mm_hidden_size,
            hidden_size=hidden_size,
            connector_args=connector_args,
        )
        
        return connector
    else:
        # Fallback to original string matching approach
        projector_type = getattr(config, "mm_projector_type", "linear")
        logger.info(f"[Legacy Projector] Using type: {projector_type}")
        
        if projector_type == "linear":
            projector = nn.Linear(mm_hidden_size, hidden_size)
        elif projector_type == "identity":
            projector = nn.Identity()
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(mm_hidden_size, hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(hidden_size, hidden_size))
                projector = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {projector_type}")
        
        return projector


# ============ Patch before importing llava ============
# This is critical: must replace build_vision_projector before llava import

def install_projector_patch():
    """
    Install projector patch
    
    Intercept import by creating a fake builder module
    """
    import importlib
    import importlib.abc
    import importlib.machinery
    
    class PatchedBuilderModule:
        """Patched builder module"""
        # Keep other content from original module
        class IdentityMap(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return x
            @property
            def config(self):
                return {"mm_projector_type": 'identity'}
        
        class SimpleResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.pre_norm = nn.LayerNorm(channels)
                self.proj = nn.Sequential(
                    nn.Linear(channels, channels),
                    nn.GELU(),
                    nn.Linear(channels, channels)
                )
            def forward(self, x):
                x = self.pre_norm(x)
                return x + self.proj(x)
        
        # Use our custom function
        build_vision_projector = staticmethod(build_custom_projector)
    
    # Inject patched module into sys.modules
    sys.modules['llava.model.multimodal_projector.builder'] = PatchedBuilderModule()
    logger.info("[Patch] Successfully installed custom projector builder")


def patch_model_arguments():
    """
    Extend ModelArguments to support custom parameters
    
    Replace original ModelArguments by creating new dataclass
    """
    import transformers
    from dataclasses import dataclass, field as dataclass_field
    
    @dataclass
    class ExtendedModelArguments:
        """Extended model arguments (including custom Connector parameters)"""
        model_name_or_path: Optional[str] = dataclass_field(default="facebook/opt-125m")
        version: Optional[str] = dataclass_field(default="v0")
        freeze_backbone: bool = dataclass_field(default=False)
        tune_mm_mlp_adapter: bool = dataclass_field(default=False)
        vision_tower: Optional[str] = dataclass_field(default=None)
        mm_vision_select_layer: Optional[int] = dataclass_field(default=-1)
        pretrain_mm_mlp_adapter: Optional[str] = dataclass_field(default=None)
        mm_projector_type: Optional[str] = dataclass_field(default='linear')
        mm_use_im_start_end: bool = dataclass_field(default=False)
        mm_use_im_patch_token: bool = dataclass_field(default=True)
        mm_patch_merge_type: Optional[str] = dataclass_field(default='flat')
        mm_vision_select_feature: Optional[str] = dataclass_field(default="patch")
        
        # ===== New custom Connector parameters =====
        use_pretrained_projector: bool = dataclass_field(
            default=False,
            metadata={
                "help": "Whether to use pretrained projector from WSI-LLaVA instead of custom connector. "
                        "When True, will use original mm_projector_type and pretrain_mm_mlp_adapter."
            }
        )
        mm_projector_path: Optional[str] = dataclass_field(
            default=None,
            metadata={
                "help": "Connector module path or registered name. "
                        "Options: 'mlp', 'qformer', or full module path. "
                        "Ignored when use_pretrained_projector=True."
            }
        )
        mm_projector_args: Optional[str] = dataclass_field(
            default=None,
            metadata={
                "help": "Connector arguments as JSON string. "
                        "Example: '{\"num_layers\": 2, \"activation\": \"gelu\"}'. "
                        "Ignored when use_pretrained_projector=True."
            }
        )
        
        # ===== Training control parameters =====
        train_connector: bool = dataclass_field(
            default=True,
            metadata={"help": "Whether to train the connector/projector."}
        )
        train_mllm: bool = dataclass_field(
            default=True,
            metadata={"help": "Whether to train the MLLM (language model)."}
        )
    
    return ExtendedModelArguments


def _find_mm_projector(model):
    """
    Robustly find mm_projector from model, compatible with normal models
    and PeftModel wrappers
    
    PeftModel wrapper attribute path:
        PeftModel.model → LlavaLlamaForCausalLM
        PeftModel.model.model → LlavaLlamaModel (mm_projector here)
    
    Normal model attribute path:
        LlavaLlamaForCausalLM.model → LlavaLlamaModel (mm_projector here)
    """
    # Method 1: Via get_model() method (LLaVA standard, compatible with PeftModel's __getattr__ proxy)
    if hasattr(model, 'get_model'):
        try:
            inner = model.get_model()
            if hasattr(inner, 'mm_projector'):
                return inner.mm_projector
        except Exception:
            pass
    
    # Method 2: PeftModel → base_model.model → get_model()
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base = model.base_model.model
        if hasattr(base, 'get_model'):
            try:
                inner = base.get_model()
                if hasattr(inner, 'mm_projector'):
                    return inner.mm_projector
            except Exception:
                pass
        if hasattr(base, 'model') and hasattr(base.model, 'mm_projector'):
            return base.model.mm_projector
    
    # Method 3: Iterate through all submodules to find
    for name, module in model.named_modules():
        if name.endswith('mm_projector'):
            return module
    
    return None


def setup_training_parameters(model, model_args):
    """
    Set model parameter trainable state according to configuration
    
    Compatible with normal LlavaLlamaForCausalLM and PeftModel (LoRA wrapped)
    
    Args:
        model: Model instance (may be PeftModel)
        model_args: Model parameter configuration
    """
    logger.info("=" * 60)
    logger.info("Setting up training parameters...")
    logger.info(f"  Train Connector: {model_args.train_connector}")
    logger.info(f"  Train MLLM: {model_args.train_mllm}")
    logger.info("=" * 60)
    
    # Multimodal component keywords to exclude
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'image_newline']
    
    # 1. Control Connector/Projector training
    connector = _find_mm_projector(model)
    if connector is not None:
        for param in connector.parameters():
            param.requires_grad = model_args.train_connector
        logger.info(f"[Connector] Set requires_grad={model_args.train_connector}")
        
        total_params = sum(p.numel() for p in connector.parameters())
        trainable_params = sum(p.numel() for p in connector.parameters() if p.requires_grad)
        logger.info(f"[Connector] Total params: {total_params:,}, Trainable: {trainable_params:,}")
    else:
        logger.warning("[Connector] mm_projector not found in model!")
    
    # 2. Control MLLM training (only affects non-multimodal, non-LoRA original LM parameters)
    mllm_params_total = 0
    mllm_params_trainable = 0
    for name, param in model.named_parameters():
        # Skip multimodal components (controlled separately above)
        if any(kw in name for kw in multimodal_keywords):
            continue
        # Skip LoRA parameters (controlled by LoRA framework)
        if 'lora_' in name:
            continue
        param.requires_grad = model_args.train_mllm
        mllm_params_total += param.numel()
        if param.requires_grad:
            mllm_params_trainable += param.numel()
    logger.info(f"[MLLM] Set requires_grad={model_args.train_mllm}")
    logger.info(f"[MLLM] Total non-multimodal/non-LoRA params: {mllm_params_total:,}, Trainable: {mllm_params_trainable:,}")
    
    # 3. Print overall trainable parameter statistics
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 60)
    logger.info(f"[Total] Model params: {total_model_params:,}")
    logger.info(f"[Total] Trainable params: {trainable_model_params:,}")
    logger.info(f"[Total] Trainable ratio: {100.0 * trainable_model_params / total_model_params:.2f}%")
    logger.info("=" * 60)


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("WSI-LLaVA Training with Custom Connector V2")
    logger.info("=" * 60)
    
    # Step 1: Install projector patch (must be before importing llava)
    install_projector_patch()
    
    # Step 2: Now safe to import llava
    import llava.train.train as train_module
    
    # Step 3: Replace ModelArguments
    ExtendedModelArguments = patch_model_arguments()
    train_module.ModelArguments = ExtendedModelArguments
    logger.info("[Patch] Successfully replaced ModelArguments")
    
    # Step 4: Hook into training process to set parameter trainable state after model creation
    # Save original train function
    original_train = train_module.train
    
    def wrapped_train(*args, **kwargs):
        """Wrapped training function to set parameters after model creation"""
        # First need to parse arguments to get model_args
        from transformers import HfArgumentParser
        from llava.train.train import DataArguments, TrainingArguments
        
        parser = HfArgumentParser((ExtendedModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        # Call original training but hook model creation
        # Here we monkey patch logic after make_supervised_data_module
        original_make_supervised = train_module.make_supervised_data_module
        model_created = [False]  # Use list to modify in closure
        
        def hooked_make_supervised(*args, **kwargs):
            result = original_make_supervised(*args, **kwargs)
            # After data module creation, model is created, set parameters now
            if not model_created[0]:
                model_created[0] = True
                # Get model from trainer
                # Note: model not yet created at this timing, need after trainer creation
            return result
        
        train_module.make_supervised_data_module = hooked_make_supervised
        
        # Call original train directly, but need to hook at different location
        # Actually better way is to set after Trainer initialization
        return original_train(*args, **kwargs)
    
    # Since LLaVA training process is complex, use more direct approach
    # Patch LlavaTrainer.__init__ before training starts
    try:
        from llava.train.llava_trainer import LLaVATrainer
        original_trainer_init = LLaVATrainer.__init__
        
        def patched_trainer_init(self, *args, **kwargs):
            original_trainer_init(self, *args, **kwargs)
            # After Trainer initialization completes, set parameter trainable state
            if hasattr(self, 'model') and hasattr(self, 'args'):
                # Get model_args from args
                # We need to get model_args from global
                if hasattr(patched_trainer_init, 'model_args_cache'):
                    setup_training_parameters(self.model, patched_trainer_init.model_args_cache)
        
        LLaVATrainer.__init__ = patched_trainer_init
        logger.info("[Patch] Successfully patched LLaVATrainer.__init__")
        
        # Cache model_args for use during Trainer initialization
        from transformers import HfArgumentParser
        from llava.train.train import DataArguments, TrainingArguments
        parser = HfArgumentParser((ExtendedModelArguments, DataArguments, TrainingArguments))
        model_args, _, _ = parser.parse_args_into_dataclasses()
        patched_trainer_init.model_args_cache = model_args
        
    except Exception as e:
        logger.warning(f"[Patch] Failed to patch LLaVATrainer: {e}")
    
    # Step 5: Run training
    logger.info("Starting training...")
    train_module.train(attn_implementation="flash_attention_2")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
