"""
WSI Feature Extraction and Target Region Summary Token Extraction
Optimized with Tensor Parallelism across 2 GPUs
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from openslide import OpenSlide
from tqdm import tqdm

# Add project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from third_party.conch.open_clip_custom import create_model_from_pretrained
from network import HSANConfig
from model.mae import HSANForMAE, SequenceParser

# ============ Global Configuration ============
CONCH_MODEL_PATH = ""
HSAN_CHECKPOINT_PATH = ""
TARGET_PATCH_SIZE = 512
PACK_GRID = 3
BATCH_SIZE = 128
WHITE_THRESHOLD = 240
DEVICE = "cuda:0"

# Tensor Parallelism Configuration
ENABLE_TENSOR_PARALLEL = True  # Set to False to disable tensor parallelism
GPU_DEVICES = ["cuda:0", "cuda:1"]  # Devices for model parallelism

# ============ Global Model Cache ============
_conch_model = None
_conch_preprocess = None
_hsan_encoder = None
_hsan_config = None
_seq_parser = None


class TensorParallelHSANEncoder(nn.Module):
    """
    Memory-Efficient Tensor Parallel Wrapper for HSAN Encoder
    Splits encoder layers across multiple GPUs to reduce per-GPU memory usage
    Uses FP16 and CPU offloading for additional memory optimization
    """
    def __init__(self, hsan_encoder, devices):
        super().__init__()
        self.hsan_encoder = hsan_encoder
        self.devices = devices
        self.config = hsan_encoder.config
        
        # Split encoder layers across devices
        encoder = self.hsan_encoder.encoder
        num_layers = len(encoder.layers)
        split_point = num_layers // 2
        
        # Device 0: input_proj + first half of layers (convert to FP16)
        encoder.input_proj = encoder.input_proj.half().to(devices[0])
        for i in range(split_point):
            encoder.layers[i] = encoder.layers[i].half().to(devices[0])
        
        # Device 1: second half of layers + norm (convert to FP16)
        for i in range(split_point, num_layers):
            encoder.layers[i] = encoder.layers[i].half().to(devices[1])
        encoder.norm = encoder.norm.half().to(devices[1])
        
        self.split_point = split_point
        print(f"✓ Memory-efficient HSAN Encoder split across {devices}")
        print(f"  - {devices[0]}: input_proj + layers[0:{split_point}] (FP16)")
        print(f"  - {devices[1]}: layers[{split_point}:{num_layers}] + norm (FP16)")
        print(f"  - Memory savings: ~50% per GPU")
    
    def forward(self, x):
        """
        Memory-efficient forward pass with tensor parallelism
        Optimizations:
        1. FP16 inference (model converted to FP16 at init)
        2. CPU offloading for attention masks
        3. Aggressive cache clearing after each layer
        4. Non-blocking transfers to overlap computation
        """
        encoder = self.hsan_encoder.encoder
        
        # Convert input to FP16 to match model dtype
        original_dtype = x.dtype
        x = x.half().to(self.devices[0])
        
        # Input projection on device 0
        x = encoder.input_proj(x)
        
        seq_len = x.size(1)
        
        # Generate mask on CPU to save GPU memory
        if encoder._cached_mask is None or encoder._cached_seq_len != seq_len:
            from network import HSANMaskGenerator, LongNetMaskGenerator
            if encoder.mask_type == "longnet":
                encoder._cached_mask = LongNetMaskGenerator.build_mask_single(
                    seq_len, encoder.args.pack_size, 'cpu'
                )
            else:
                encoder._cached_mask = HSANMaskGenerator.build_mask(
                    seq_len, encoder.args.pack_size, 'cpu'
                )
            encoder._cached_seq_len = seq_len
        
        # First half of layers on device 0
        for i in range(self.split_point):
            # Only transfer mask to GPU when needed (non-blocking)
            # Mask stays in its original dtype (typically bool or float32)
            mask_device0 = encoder._cached_mask.to(self.devices[0], non_blocking=True)
            x = encoder.layers[i](x, mask_device0)
            del mask_device0
            
            # Aggressive cache clearing after each layer
            torch.cuda.empty_cache()
        
        # Clear device 0 before transfer
        torch.cuda.empty_cache()
        
        # Transfer to device 1 (non-blocking)
        x = x.to(self.devices[1], non_blocking=True)
        
        # Second half of layers on device 1
        for i in range(self.split_point, len(encoder.layers)):
            # Only transfer mask to GPU when needed (non-blocking)
            # Mask stays in its original dtype (typically bool or float32)
            mask_device1 = encoder._cached_mask.to(self.devices[1], non_blocking=True)
            x = encoder.layers[i](x, mask_device1)
            del mask_device1
            
            # Aggressive cache clearing after each layer
            torch.cuda.empty_cache()
        
        # Final norm on device 1
        x = encoder.norm(x)
        
        # Convert back to original dtype
        x = x.to(original_dtype)
        
        return x


def _load_conch_model():
    """Load CONCH model (lazy loading)"""
    global _conch_model, _conch_preprocess
    if _conch_model is None:
        print("Loading CONCH model...")
        _conch_model, _conch_preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=CONCH_MODEL_PATH
        )
        _conch_model = _conch_model.to(DEVICE).eval()
    return _conch_model, _conch_preprocess


def _load_hsan_model():
    """Load HSAN model (lazy loading) with optional tensor parallelism"""
    global _hsan_encoder, _hsan_config, _seq_parser
    if _hsan_encoder is None:
        print("Loading HSAN model...")
        
        # Check GPU availability
        if ENABLE_TENSOR_PARALLEL:
            available_gpus = [torch.cuda.is_available() and i < torch.cuda.device_count() 
                            for i, _ in enumerate(GPU_DEVICES)]
            if not all(available_gpus[:2]):
                print(f"Warning: Only {sum(available_gpus)} GPU(s) available. Disabling tensor parallelism.")
                use_parallel = False
            else:
                use_parallel = True
                print(f"Using tensor parallelism across {GPU_DEVICES[:2]}")
        else:
            use_parallel = False
        
        # Load checkpoint on CPU first to save memory
        checkpoint = torch.load(HSAN_CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        _hsan_config = checkpoint.get('config', HSANConfig())
        
        # Create model (on CPU)
        hsan_model = HSANForMAE(_hsan_config, use_gradient_checkpointing=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('encoder_state_dict'))
        hsan_model.load_state_dict(state_dict, strict=False)
        hsan_model.eval()
        
        # Apply tensor parallelism or move to single device
        if use_parallel:
            _hsan_encoder = TensorParallelHSANEncoder(hsan_model, GPU_DEVICES[:2])
        else:
            # Convert to FP16 for memory efficiency
            _hsan_encoder = hsan_model.half().to(DEVICE)
            print(f"HSAN model loaded on {DEVICE} (FP16 for memory efficiency)")
            print(f"  - Memory savings: ~50%")
        
        _seq_parser = SequenceParser(_hsan_config.pack_size)
        
        # Clear checkpoint from memory
        del checkpoint
        torch.cuda.empty_cache()
        
    return _hsan_encoder, _hsan_config, _seq_parser


def _encode_batch(model, preprocess, images):
    """Encode a batch of images"""
    tensors = torch.stack([preprocess(img) for img in images]).to(DEVICE)
    with torch.inference_mode():
        features = model.encode_image(tensors, proj_contrast=True, normalize=True)
    return features.cpu()


def _is_white_pack(patches):
    """Check if pack is white background"""
    white_count = sum(1 for p in patches if np.array(p).mean() > WHITE_THRESHOLD)
    return white_count > len(patches) / 2 + 1


def _get_pack_thumbnail(patches):
    """Generate pack thumbnail"""
    pack_img = Image.new('RGB', (TARGET_PATCH_SIZE * PACK_GRID, TARGET_PATCH_SIZE * PACK_GRID))
    for idx, patch in enumerate(patches):
        row, col = idx // PACK_GRID, idx % PACK_GRID
        pack_img.paste(patch, (col * TARGET_PATCH_SIZE, row * TARGET_PATCH_SIZE))
    return pack_img.resize((TARGET_PATCH_SIZE, TARGET_PATCH_SIZE), Image.BICUBIC)


def extract_region_features(wsi_path: str, targets: dict):
    """
    Extract Summary Token features from target regions in WSI
    Optimized with tensor parallelism for memory efficiency
    
    Args:
        wsi_path: Path to WSI file
        targets: Target region dictionary
            key: "[[x1,y1],[x2,y2]]" or ((x1,y1),(x2,y2)) - top-left and bottom-right coordinates
            value: Output path "xx/xx/xx.pt"
    """
    # Load models (HSAN will be split across GPUs if tensor parallelism is enabled)
    conch_model, preprocess = _load_conch_model()
    hsan_encoder, hsan_config, seq_parser = _load_hsan_model()
    
    slide = OpenSlide(wsi_path)
    w, h = slide.dimensions
    print(f"Processing WSI: {wsi_path}, Size: {w}x{h}")
    
    # Calculate grid parameters
    pack_pixels = TARGET_PATCH_SIZE * PACK_GRID
    num_packs_x = max(1, w // pack_pixels)
    num_packs_y = max(1, h // pack_pixels)
    actual_patch_w = w // (num_packs_x * PACK_GRID)
    actual_patch_h = h // (num_packs_y * PACK_GRID)
    pack_stride_x = actual_patch_w * PACK_GRID
    pack_stride_y = actual_patch_h * PACK_GRID
    
    print(f"Grid: {num_packs_x}x{num_packs_y} packs, Patch size: {actual_patch_w}x{actual_patch_h}")
    
    # 1. Encode global thumbnail
    global_thumb = slide.get_thumbnail((TARGET_PATCH_SIZE, TARGET_PATCH_SIZE))
    global_feature = _encode_batch(conch_model, preprocess, [global_thumb])[0].unsqueeze(0)
    del global_thumb  # Immediately release memory
    
    # 2. Collect pack coordinates only (do NOT store images to save memory)
    valid_pack_coords = []  # [(pack_x, pack_y, px_idx, py_idx)]
    
    print("Scanning for valid packs...")
    for py_idx in range(num_packs_y):
        for px_idx in range(num_packs_x):
            pack_x = px_idx * pack_stride_x
            pack_y = py_idx * pack_stride_y
            
            # Read all patches in pack to check validity
            patches = []
            for row in range(PACK_GRID):
                for col in range(PACK_GRID):
                    x = pack_x + col * actual_patch_w
                    y = pack_y + row * actual_patch_h
                    region = slide.read_region((x, y), 0, (actual_patch_w, actual_patch_h)).convert("RGB")
                    if (actual_patch_w, actual_patch_h) != (TARGET_PATCH_SIZE, TARGET_PATCH_SIZE):
                        region = region.resize((TARGET_PATCH_SIZE, TARGET_PATCH_SIZE), Image.BICUBIC)
                    patches.append(region)
            
            if not _is_white_pack(patches):
                valid_pack_coords.append((pack_x, pack_y, px_idx, py_idx))
            
            # Immediately release patches
            del patches
    
    print(f"Valid packs: {len(valid_pack_coords)}/{num_packs_x * num_packs_y}")
    
    if not valid_pack_coords:
        print("No valid packs found!")
        slide.close()
        return
    
    # 3. Encode patches one pack at a time to minimize memory usage
    encoded_features = [global_feature]
    ENCODE_BATCH_SIZE = min(BATCH_SIZE, 32)  # Use smaller batch size for memory efficiency
    
    for pack_idx in tqdm(range(len(valid_pack_coords)), desc="Encoding packs"):
        pack_x, pack_y, px_idx, py_idx = valid_pack_coords[pack_idx]
        
        # Read patches for current pack only
        patches = []
        for row in range(PACK_GRID):
            for col in range(PACK_GRID):
                x = pack_x + col * actual_patch_w
                y = pack_y + row * actual_patch_h
                region = slide.read_region((x, y), 0, (actual_patch_w, actual_patch_h)).convert("RGB")
                if (actual_patch_w, actual_patch_h) != (TARGET_PATCH_SIZE, TARGET_PATCH_SIZE):
                    region = region.resize((TARGET_PATCH_SIZE, TARGET_PATCH_SIZE), Image.BICUBIC)
                patches.append(region)
        
        # Generate thumbnail for current pack
        thumbnail = _get_pack_thumbnail(patches)
        pack_images = patches + [thumbnail]
        
        # Encode current pack
        pack_encoded = []
        for i in range(0, len(pack_images), ENCODE_BATCH_SIZE):
            batch = pack_images[i:i+ENCODE_BATCH_SIZE]
            pack_encoded.append(_encode_batch(conch_model, preprocess, batch))
        pack_encoded = torch.cat(pack_encoded, dim=0)
        
        # Add to feature sequence: patches + summary
        encoded_features.append(pack_encoded[:PACK_GRID * PACK_GRID])  # patches
        encoded_features.append(pack_encoded[PACK_GRID * PACK_GRID:])  # summary
        
        # Immediately release current pack data
        del patches, thumbnail, pack_images, pack_encoded
        
        # Periodic GPU cache clearing
        if (pack_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    slide.close()
    
    # 4. Concatenate all features - use CPU if GPU memory is insufficient
    try:
        features = torch.cat(encoded_features, dim=0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("GPU memory insufficient for feature concatenation, using CPU storage")
            torch.cuda.empty_cache()
            encoded_features_cpu = [f.cpu() for f in encoded_features]
            features = torch.cat(encoded_features_cpu, dim=0)
            del encoded_features_cpu
        else:
            raise e
    
    del encoded_features
    print(f"Feature sequence shape: {features.shape}")
    
    # 5. Extract Summary Tokens with memory-efficient HSAN encoding
    print("Processing HSAN encoding (memory-efficient mode)...")
    
    with torch.no_grad():
        # Clear GPU cache before HSAN processing
        torch.cuda.empty_cache()
        
        # Determine input device
        if isinstance(hsan_encoder, TensorParallelHSANEncoder):
            input_device = GPU_DEVICES[0]
        else:
            input_device = DEVICE
        
        # Transfer features to device (will handle CPU->GPU transfer if needed)
        if features.device.type == 'cpu':
            print(f"Transferring features from CPU to {input_device}...")
        
        x = features.unsqueeze(0)
        
        # Forward pass (handles device placement and memory optimization internally)
        if isinstance(hsan_encoder, TensorParallelHSANEncoder):
            # TensorParallelHSANEncoder handles FP16 conversion internally
            encoded_seq = hsan_encoder(x)
        else:
            # Model is already FP16, convert input to match
            x = x.half().to(input_device)
            encoded_seq = hsan_encoder.encoder(x)
            # Convert output back to FP32 for consistency
            encoded_seq = encoded_seq.float()
        
        # Parse sequence structure
        parsed = seq_parser.parse(features.shape[0])
        summary_indices = parsed['summary_indices']
        
        # Extract summary tokens (move to CPU for storage)
        summary_tokens = encoded_seq[0, summary_indices, :].cpu()
        
        # Clean up intermediate tensors
        del x, encoded_seq
    
    print(f"Extracted {len(summary_indices)} summary tokens")
    
    # Final GPU cache clearing
    torch.cuda.empty_cache()
    
    # 6. Build pack coordinate mapping
    # summary_indices[i] corresponds to valid_pack_coords[i]'s summary token
    
    # 7. Process each target region
    for region_key, output_path in targets.items():
        # Parse coordinates
        if isinstance(region_key, str):
            import ast
            coords = ast.literal_eval(region_key)
        else:
            coords = region_key
        
        (x1, y1), (x2, y2) = coords
        
        # Find all packs covering this region (can expand, cannot shrink)
        region_pack_indices = []
        for i, (pack_x, pack_y, _, _) in enumerate(valid_pack_coords):
            pack_x2 = pack_x + pack_stride_x
            pack_y2 = pack_y + pack_stride_y
            
            # Check if pack intersects with target region
            if not (pack_x2 <= x1 or pack_x >= x2 or pack_y2 <= y1 or pack_y >= y2):
                region_pack_indices.append(i)
        
        if not region_pack_indices:
            print(f"Warning: No packs found for region {region_key}, skipping")
            continue
        
        # Extract corresponding summary tokens
        region_summary_tokens = summary_tokens[region_pack_indices]
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save({
            'features': region_summary_tokens,
            'pack_indices': region_pack_indices,
            'region_coords': coords,
            'num_tokens': len(region_pack_indices)
        }, output_path)
        
        print(f"Saved {len(region_pack_indices)} tokens to {output_path}")


if __name__ == "__main__":
    # Example usage
    wsi_path = ""
    targets = {
        "[[0,0],[5000,5000]]": "./output/region1.pt",
        "[[10000,10000],[15000,15000]]": "./output/region2.pt",
    }
    extract_region_features(wsi_path, targets)
