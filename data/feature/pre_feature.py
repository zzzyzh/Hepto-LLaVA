"""
WSI Feature Preprocessing Program
- 9 Patches (3x3) form one Pack
- Pack count adapts to WSI dimensions
- Packs with over half white background are discarded
- Row-major order arrangement
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from openslide import OpenSlide
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from third_party.conch.open_clip_custom import create_model_from_pretrained


class WSIFeatureExtractor:
    def __init__(self, model_path: str, target_patch_size: int = 512, 
                 pack_grid: int = 3, batch_size: int = 64, white_threshold: float = 240):
        self.target_patch_size = target_patch_size
        self.pack_grid = pack_grid
        self.batch_size = batch_size
        self.white_threshold = white_threshold
        
        print(f"Loading CONCH model...")
        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=model_path
        )
        self.model = self.model.cuda().eval()
        print("Model loaded.")

    def compute_grid_params(self, slide: OpenSlide):
        w, h = slide.dimensions
        pack_pixels = self.target_patch_size * self.pack_grid
        
        num_packs_x = max(1, w // pack_pixels)
        num_packs_y = max(1, h // pack_pixels)
        
        actual_patch_w = w // (num_packs_x * self.pack_grid)
        actual_patch_h = h // (num_packs_y * self.pack_grid)
        
        return (actual_patch_w, actual_patch_h), num_packs_x, num_packs_y

    def is_white_pack(self, patches: list) -> bool:
        white_count = sum(1 for p in patches if np.array(p).mean() > self.white_threshold)
        return white_count > len(patches) / 2

    def read_pack(self, slide: OpenSlide, pack_x: int, pack_y: int, 
                  patch_size: tuple) -> list:
        patches = []
        pw, ph = patch_size
        
        for row in range(self.pack_grid):
            for col in range(self.pack_grid):
                x = pack_x + col * pw
                y = pack_y + row * ph
                region = slide.read_region((x, y), 0, (pw, ph)).convert("RGB")
                if (pw, ph) != (self.target_patch_size, self.target_patch_size):
                    region = region.resize((self.target_patch_size, self.target_patch_size), Image.BICUBIC)
                patches.append(region)
        return patches

    def encode_batch(self, patches: list) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(p) for p in patches]).cuda()
        with torch.inference_mode():
            features = self.model.encode_image(tensors, proj_contrast=True, normalize=True)
        return features.cpu()
    
    def get_global_thumbnail(self, slide: OpenSlide, target_size: int = 512) -> Image.Image:
        w, h = slide.dimensions
        
        try:
            thumbnail = slide.get_thumbnail((target_size, target_size))
        except:
            print("Failed to get thumbnail using OpenSlide.get_thumbnail()")
            level = slide.level_count - 1
            level_w, level_h = slide.level_dimensions[level]
            thumbnail = slide.read_region((0, 0), level, (level_w, level_h)).convert("RGB")
            thumbnail = thumbnail.resize((target_size, target_size), Image.BICUBIC)
        
        return thumbnail
    
    def get_pack_thumbnail(self, patches: list) -> Image.Image:
        patch_size = self.target_patch_size
        pack_img = Image.new('RGB', (patch_size * self.pack_grid, patch_size * self.pack_grid))
        
        for idx, patch in enumerate(patches):
            row = idx // self.pack_grid
            col = idx % self.pack_grid
            x = col * patch_size
            y = row * patch_size
            pack_img.paste(patch, (x, y))
        
        pack_thumbnail = pack_img.resize((patch_size, patch_size), Image.BICUBIC)
        return pack_thumbnail

    def process(self, slide_path: str) -> tuple:
        slide = OpenSlide(slide_path)
        print(f"Slide: {slide_path}, Size: {slide.dimensions}")
        
        print("Encoding global thumbnail...")
        global_thumbnail = self.get_global_thumbnail(slide)
        global_feature = self.encode_batch([global_thumbnail])[0].unsqueeze(0)
        
        patch_size, num_packs_x, num_packs_y = self.compute_grid_params(slide)
        pack_stride_x = patch_size[0] * self.pack_grid
        pack_stride_y = patch_size[1] * self.pack_grid
        
        print(f"Grid: {num_packs_x}x{num_packs_y} packs, Patch size: {patch_size}")
        
        valid_packs = []
        valid_coords = []
        
        for py_idx in range(num_packs_y):
            for px_idx in range(num_packs_x):
                pack_x = px_idx * pack_stride_x
                pack_y = py_idx * pack_stride_y
                
                patches = self.read_pack(slide, pack_x, pack_y, patch_size)
                
                if not self.is_white_pack(patches):
                    pack_thumbnail = self.get_pack_thumbnail(patches)
                    valid_packs.append((patches, pack_thumbnail, (pack_x, pack_y)))
                    valid_coords.append((pack_x, pack_y))
        
        slide.close()
        print(f"Valid packs: {len(valid_packs)}/{num_packs_x * num_packs_y}")
        
        if not valid_packs:
            return torch.empty(0, 512), [], {}
        
        print("Encoding features in network order...")
        all_features = [global_feature]
        
        all_images_to_encode = []
        pack_boundaries = []
        
        for patches, pack_thumbnail, _ in valid_packs:
            start_idx = len(all_images_to_encode)
            all_images_to_encode.extend(patches)
            all_images_to_encode.append(pack_thumbnail)
            end_idx = len(all_images_to_encode)
            pack_boundaries.append((start_idx, end_idx))
        
        encoded_features = []
        for i in tqdm(range(0, len(all_images_to_encode), self.batch_size), desc="Encoding"):
            batch = all_images_to_encode[i:i+self.batch_size]
            encoded_features.append(self.encode_batch(batch))
        encoded_features = torch.cat(encoded_features, dim=0)
        
        for start_idx, end_idx in pack_boundaries:
            pack_patches = encoded_features[start_idx:end_idx-1]
            pack_summary = encoded_features[end_idx-1:end_idx]
            all_features.append(pack_patches)
            all_features.append(pack_summary)
        
        features = torch.cat(all_features, dim=0)
        
        config = {
            'num_packs': len(valid_coords),
            'num_patches': len(valid_coords) * self.pack_grid,
            'pack_grid': self.pack_grid,
            'target_patch_size': self.target_patch_size,
            'actual_patch_size': patch_size,
            'seq_len': features.shape[0]
        }
        
        print(f"Output features shape: {features.shape}")
        print(f"Sequence structure: [Global(1)] + [Pack1_patches(9) + Pack1_summary(1)] * {len(valid_coords)}")
        
        return features, valid_coords, config


def main():
    parser = argparse.ArgumentParser(description="WSI Feature Extractor")
    parser.add_argument("--input", "-i", type=str, required=True, help="WSI file or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--model", "-m", type=str, 
                        default="./models/CONCH/pytorch_model.bin",
                        help="Path to CONCH model file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=512)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    extractor = WSIFeatureExtractor(
        model_path=args.model,
        target_patch_size=args.patch_size,
        batch_size=args.batch_size
    )
    
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi'))]
    else:
        files = [args.input]
    
    for slide_path in files:
        print(f"\n{'='*50}")
        try:
            features, coords, config = extractor.process(slide_path)
            
            name = os.path.splitext(os.path.basename(slide_path))[0]
            output_path = os.path.join(args.output, f"{name}.pt")
            torch.save({
                'features': features,
                'pack_coords': coords,
                'config': config
            }, output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone.")


if __name__ == "__main__":
    main()
