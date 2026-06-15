#!/usr/bin/env python3
"""
Preprocess feature files
Convert 512-dim features to 1024-dim (zero-padding) or other formats
"""

import torch
import os
import sys
import json
from pathlib import Path
from tqdm import tqdm


def load_qa_data(jsonl_path):
    """Load QA data from JSONL file"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_image_paths(qa_data):
    """Extract all image paths from QA data"""
    image_paths = set()
    for sample in qa_data:
        if 'image' in sample:
            image_paths.add(sample['image'])
    return sorted(image_paths)


def process_feature_file(input_path, output_path):
    """Process single feature file, zero-pad to 1024 dimensions"""
    try:
        # Load features
        data = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Extract feature tensor
        if isinstance(data, dict):
            if 'features' in data:
                features = data['features']
            elif 'summary_tokens' in data:
                features = data['summary_tokens']
            else:
                print(f"  ⚠️  Unknown dict format, skip. Keys: {list(data.keys())}")
                return False
        else:
            features = data
        
        # Convert to tensor
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).float()
        else:
            features = features.float()
        
        # Check dimensions
        if features.dim() == 1:
            features = features.unsqueeze(0)  # (dim,) -> (1, dim)
        
        num_tokens, embed_dim = features.shape
        
        # If already 1024-dim, save directly
        if embed_dim == 1024:
            torch.save(features, output_path)
            return True
        
        # Zero-pad to 1024 dimensions
        if embed_dim == 512:
            pad_size = 1024 - embed_dim
            padding = torch.zeros(num_tokens, pad_size, dtype=features.dtype)
            features_padded = torch.cat([features, padding], dim=1)
            
            # Save
            torch.save(features_padded, output_path)
            return True
        else:
            print(f"  ⚠️  Unsupported feature dimension: {embed_dim}, skip")
            return False
            
    except Exception as e:
        print(f"  ✗ Processing failed: {str(e)}")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: python preprocess_hepatovqa_features.py <input_dir> <output_dir> <qa_file>")
        print("Example: python preprocess_hepatovqa_features.py ./data/input ./data/output ./data/qa_merged.jsonl")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    qa_file = Path(sys.argv[3])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load QA data
    print("Loading QA data...")
    qa_data = load_qa_data(qa_file)
    print(f"Found {len(qa_data)} samples")
    
    # Extract image paths
    image_paths = extract_image_paths(qa_data)
    print(f"Found {len(image_paths)} unique feature files")
    
    print("\n" + "=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70 + "\n")
    
    # Process all files
    success_count = 0
    for i, rel_path in enumerate(tqdm(image_paths, desc="Processing features"), 1):
        input_path = input_dir / rel_path
        output_path = output_dir / rel_path
        
        # Create output subdirectory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if input file exists
        if not input_path.exists():
            print(f"[{i}/{len(image_paths)}] File not found: {input_path}")
            continue
        
        # Process file
        if process_feature_file(input_path, output_path):
            success_count += 1
    
    print(f"\n{'=' * 70}")
    print(f"Processing complete! Success: {success_count}/{len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
