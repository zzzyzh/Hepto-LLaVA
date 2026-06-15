"""
WSI Data Augmentation Program
- Generates 9 .pt files for each WSI:
  1. Original complete WSI (original)
  2. Four quadrants: top-left(tl), top-right(tr), bottom-left(bl), bottom-right(br)
  3. Four flipped versions: tl_flip, tr_flip, bl_flip, br_flip
- Reuses WSIFeatureExtractor from pre_feature.py
"""

import os
import sys
import argparse
import torch
from openslide import OpenSlide
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pre_feature import WSIFeatureExtractor


class CroppedFlippedSlide:
    """
    Wrapper for OpenSlide supporting region cropping and flipping
    Makes it appear as an independent WSI
    """
    def __init__(self, slide, crop_region=None, flip_h=False, flip_v=False):
        """
        Args:
            slide: OpenSlide object
            crop_region: (x, y, width, height) crop region, None means no cropping
            flip_h: Whether to flip horizontally
            flip_v: Whether to flip vertically
        """
        self.slide = slide
        self.crop_region = crop_region
        self.flip_h = flip_h
        self.flip_v = flip_v
        
        self.level_count = slide.level_count
        self.level_dimensions = slide.level_dimensions
        self.level_downsamples = slide.level_downsamples
    
    @property
    def dimensions(self):
        if self.crop_region:
            return (self.crop_region[2], self.crop_region[3])
        return self.slide.dimensions
    
    def read_region(self, location, level, size):
        x, y = location
        
        if self.crop_region:
            crop_x, crop_y, crop_w, crop_h = self.crop_region
            
            if self.flip_h:
                x = crop_w - x - size[0]
            
            if self.flip_v:
                y = crop_h - y - size[1]
            
            x += crop_x
            y += crop_y
        
        region = self.slide.read_region((x, y), level, size).convert("RGB")
        
        if self.flip_h:
            region = region.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v:
            region = region.transpose(Image.FLIP_TOP_BOTTOM)
        
        return region
    
    def get_thumbnail(self, size):
        if self.crop_region:
            x, y, w, h = self.crop_region
            level = self.level_count - 1
            downsample = self.level_downsamples[level]
            level_w = int(w / downsample)
            level_h = int(h / downsample)
            level_x = int(x / downsample)
            level_y = int(y / downsample)
            
            thumbnail = self.slide.read_region((level_x * int(downsample), level_y * int(downsample)), 
                                               level, (level_w, level_h)).convert("RGB")
        else:
            thumbnail = self.slide.get_thumbnail(size)
        
        if self.flip_h:
            thumbnail = thumbnail.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v:
            thumbnail = thumbnail.transpose(Image.FLIP_TOP_BOTTOM)
        
        thumbnail = thumbnail.resize(size, Image.BICUBIC)
        return thumbnail
    
    def close(self):
        pass


class AugmentedWSIProcessor:
    def __init__(self, extractor: WSIFeatureExtractor, save_vis: bool = False, vis_dir: str = None):
        self.extractor = extractor
        self.save_vis = save_vis
        self.vis_dir = vis_dir
        
        if self.save_vis and self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
    
    def get_quadrants(self, slide: OpenSlide):
        w, h = slide.dimensions
        half_w = w // 2
        half_h = h // 2
        
        quadrants = [
            ("tl", (0, 0, half_w, half_h)),
            ("tr", (half_w, 0, half_w, half_h)),
            ("bl", (0, half_h, half_w, half_h)),
            ("br", (half_w, half_h, half_w, half_h)),
        ]
        return quadrants
    
    def save_thumbnail(self, wrapped_slide, base_name: str, aug_type: str, size: int = 512):
        if not self.vis_dir:
            return
        
        try:
            thumbnail = wrapped_slide.get_thumbnail((size, size))
            thumbnail_path = os.path.join(self.vis_dir, f"{base_name}_{aug_type}.png")
            thumbnail.save(thumbnail_path)
            print(f"  Saved thumbnail: {thumbnail_path}")
        except Exception as e:
            print(f"  Warning: Failed to save thumbnail for {aug_type}: {e}")
    
    def process_augmented(self, slide_path: str, output_dir: str):
        """Process a WSI and generate 9 augmented versions with resume support"""
        slide = OpenSlide(slide_path)
        base_name = os.path.splitext(os.path.basename(slide_path))[0]
        
        print(f"\n{'='*60}")
        print(f"Processing: {slide_path}")
        print(f"Dimensions: {slide.dimensions}")
        
        augment_versions = [
            'original',
            'tl', 'tr', 'bl', 'br',
            'tl_flip', 'tr_flip', 'bl_flip', 'br_flip'
        ]
        
        existing_versions = []
        pending_versions = []
        for version in augment_versions:
            output_path = os.path.join(output_dir, f"{base_name}_{version}.pt")
            if os.path.exists(output_path):
                existing_versions.append(version)
            else:
                pending_versions.append(version)
        
        if existing_versions:
            print(f"Already exist: {len(existing_versions)}/9 versions - {existing_versions}")
        
        if not pending_versions:
            print(f"All 9 versions already exist, skipping.")
            slide.close()
            return [(v, os.path.join(output_dir, f"{base_name}_{v}.pt")) for v in augment_versions]
        
        print(f"Will generate {len(pending_versions)}/9 versions: {pending_versions}")
        
        results = []
        
        if 'original' in pending_versions:
            print(f"\n[1/9] Processing original WSI...")
            try:
                wrapped_slide = CroppedFlippedSlide(slide)
                features, coords, config = self.process_wrapped_slide(wrapped_slide, slide_path)
                
                output_path = os.path.join(output_dir, f"{base_name}_original.pt")
                torch.save({
                    'features': features,
                    'pack_coords': coords,
                    'config': config,
                    'augmentation': 'original'
                }, output_path)
                print(f"Saved: {output_path}")
                results.append(('original', output_path))
                
                if self.save_vis:
                    self.save_thumbnail(wrapped_slide, base_name, 'original')
            except Exception as e:
                print(f"Error processing original: {e}")
        else:
            print(f"\n[1/9] Skipping original WSI (already exists)")
            results.append(('original', os.path.join(output_dir, f"{base_name}_original.pt")))
        
        quadrants = self.get_quadrants(slide)
        for idx, (quad_name, crop_region) in enumerate(quadrants, start=2):
            if quad_name in pending_versions:
                print(f"\n[{idx}/9] Processing quadrant: {quad_name} {crop_region}...")
                try:
                    wrapped_slide = CroppedFlippedSlide(slide, crop_region=crop_region)
                    features, coords, config = self.process_wrapped_slide(wrapped_slide, slide_path)
                    
                    output_path = os.path.join(output_dir, f"{base_name}_{quad_name}.pt")
                    torch.save({
                        'features': features,
                        'pack_coords': coords,
                        'config': config,
                        'augmentation': quad_name,
                        'crop_region': crop_region
                    }, output_path)
                    print(f"Saved: {output_path}")
                    results.append((quad_name, output_path))
                    
                    if self.save_vis:
                        self.save_thumbnail(wrapped_slide, base_name, quad_name)
                except Exception as e:
                    print(f"Error processing {quad_name}: {e}")
            else:
                print(f"\n[{idx}/9] Skipping quadrant: {quad_name} (already exists)")
                results.append((quad_name, os.path.join(output_dir, f"{base_name}_{quad_name}.pt")))
        
        flip_configs = [
            ("tl_flip", quadrants[0][1], True, True),
            ("tr_flip", quadrants[1][1], True, False),
            ("bl_flip", quadrants[2][1], False, True),
            ("br_flip", quadrants[3][1], True, True),
        ]
        
        for idx, (flip_name, crop_region, flip_v, flip_h) in enumerate(flip_configs, start=6):
            if flip_name in pending_versions:
                print(f"\n[{idx}/9] Processing flipped quadrant: {flip_name} (flip_h={flip_h}, flip_v={flip_v})...")
                try:
                    wrapped_slide = CroppedFlippedSlide(slide, crop_region=crop_region, 
                                                       flip_h=flip_h, flip_v=flip_v)
                    features, coords, config = self.process_wrapped_slide(wrapped_slide, slide_path)
                    
                    output_path = os.path.join(output_dir, f"{base_name}_{flip_name}.pt")
                    torch.save({
                        'features': features,
                        'pack_coords': coords,
                        'config': config,
                        'augmentation': flip_name,
                        'crop_region': crop_region,
                        'flip_h': flip_h,
                        'flip_v': flip_v
                    }, output_path)
                    print(f"Saved: {output_path}")
                    results.append((flip_name, output_path))
                    
                    # 保存缩略图
                    if self.save_vis:
                        self.save_thumbnail(wrapped_slide, base_name, flip_name)
                except Exception as e:
                    print(f"Error processing {flip_name}: {e}")
            else:
                print(f"\n[{idx}/9] Skipping flipped quadrant: {flip_name} (already exists)")
                results.append((flip_name, os.path.join(output_dir, f"{base_name}_{flip_name}.pt")))
        
        slide.close()
        
        print(f"\n{'='*60}")
        print(f"Completed {base_name}: {len(results)}/9 versions generated")
        return results
    
    def process_wrapped_slide(self, wrapped_slide, original_path):
        print(f"Virtual WSI dimensions: {wrapped_slide.dimensions}")
        
        global_thumbnail = self.extractor.get_global_thumbnail(wrapped_slide)
        global_feature = self.extractor.encode_batch([global_thumbnail])[0].unsqueeze(0)
        
        patch_size, num_packs_x, num_packs_y = self.extractor.compute_grid_params(wrapped_slide)
        pack_stride_x = patch_size[0] * self.extractor.pack_grid
        pack_stride_y = patch_size[1] * self.extractor.pack_grid
        
        print(f"Grid: {num_packs_x}x{num_packs_y} packs, Patch size: {patch_size}")
        
        valid_packs = []
        valid_coords = []
        
        for py_idx in range(num_packs_y):
            for px_idx in range(num_packs_x):
                pack_x = px_idx * pack_stride_x
                pack_y = py_idx * pack_stride_y
                
                patches = self.extractor.read_pack(wrapped_slide, pack_x, pack_y, patch_size)
                
                if not self.extractor.is_white_pack(patches):
                    pack_thumbnail = self.extractor.get_pack_thumbnail(patches)
                    valid_packs.append((patches, pack_thumbnail, (pack_x, pack_y)))
                    valid_coords.append((pack_x, pack_y))
        
        print(f"Valid packs: {len(valid_packs)}/{num_packs_x * num_packs_y}")
        
        if not valid_packs:
            return torch.empty(0, 512), [], {}
        
        print("Encoding features...")
        all_features = [global_feature]
        
        all_images_to_encode = []
        pack_boundaries = []
        
        for patches, pack_thumbnail, _ in valid_packs:
            start_idx = len(all_images_to_encode)
            all_images_to_encode.extend(patches)
            all_images_to_encode.append(pack_thumbnail)
            end_idx = len(all_images_to_encode)
            pack_boundaries.append((start_idx, end_idx))
        
        from tqdm import tqdm
        encoded_features = []
        for i in tqdm(range(0, len(all_images_to_encode), self.extractor.batch_size), 
                     desc="Encoding", leave=False):
            batch = all_images_to_encode[i:i+self.extractor.batch_size]
            encoded_features.append(self.extractor.encode_batch(batch))
        encoded_features = torch.cat(encoded_features, dim=0)
        
        for start_idx, end_idx in pack_boundaries:
            pack_patches = encoded_features[start_idx:end_idx-1]
            pack_summary = encoded_features[end_idx-1:end_idx]
            all_features.append(pack_patches)
            all_features.append(pack_summary)
        
        features = torch.cat(all_features, dim=0)
        
        config = {
            'num_packs': len(valid_coords),
            'num_patches': len(valid_coords) * self.extractor.pack_grid,
            'pack_grid': self.extractor.pack_grid,
            'target_patch_size': self.extractor.target_patch_size,
            'actual_patch_size': patch_size,
            'seq_len': features.shape[0]
        }
        
        print(f"Output features shape: {features.shape}")
        
        return features, valid_coords, config


def main():
    parser = argparse.ArgumentParser(description="WSI Augmented Feature Extractor")
    parser.add_argument("--input", "-i", type=str, required=True, help="WSI file or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--model", "-m", type=str, 
                        default="./models/CONCH/pytorch_model.bin",
                        help="Path to CONCH model file")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size")
    parser.add_argument("--save-vis", action="store_true", help="Save augmentation thumbnails")
    parser.add_argument("--vis-dir", type=str, default=None, 
                        help="Thumbnail directory (default: output_dir/thumbnails)")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    vis_dir = args.vis_dir if args.vis_dir else os.path.join(args.output, "thumbnails")
    
    base_extractor = WSIFeatureExtractor(
        model_path=args.model,
        target_patch_size=args.patch_size,
        batch_size=args.batch_size
    )
    
    augmented_processor = AugmentedWSIProcessor(
        base_extractor, 
        save_vis=args.save_vis, 
        vis_dir=vis_dir if args.save_vis else None
    )
    
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi'))]
    else:
        files = [args.input]
    
    print(f"Found {len(files)} WSI file(s) to process")
    
    all_results = []
    for slide_path in files:
        try:
            results = augmented_processor.process_augmented(slide_path, args.output)
            all_results.append((slide_path, results))
        except Exception as e:
            print(f"Error processing {slide_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Processing Summary:")
    for slide_path, results in all_results:
        print(f"{os.path.basename(slide_path)}: {len(results)}/9 versions")
    print("="*60)
    print("Done.")


if __name__ == "__main__":
    main()

