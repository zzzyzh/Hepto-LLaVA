"""
Convert all session inference_results.json to LLaVA fine-tuning data format
and move feature files to output directory
"""
import json
import sys
import shutil
from pathlib import Path

# Define mapping between summary fields and corresponding image_path and id
FIELD_MAPPINGS = [
    # (summary_key, image_path_key, id_suffix)
    ("wsi_summary", "wsi_thumbnail_path", ""),
    ("cluster_0_summary", "cluster_0_image_path", "_cluster_0"),
    ("cluster_0_roi_10x_summary", "cluster_0_roi_10x_image_path", "_cluster_0_10x"),
    ("cluster_0_roi_20x_summary", "cluster_0_roi_20x_image_path", "_cluster_0_20x"),
    ("cluster_1_summary", "cluster_1_image_path", "_cluster_1"),
    ("cluster_1_roi_10x_summary", "cluster_1_roi_10x_image_path", "_cluster_1_10x"),
    ("cluster_1_roi_20x_summary", "cluster_1_roi_20x_image_path", "_cluster_1_20x"),
    ("cluster_2_summary", "cluster_2_image_path", "_cluster_2"),
    ("cluster_2_roi_10x_summary", "cluster_2_roi_10x_image_path", "_cluster_2_10x"),
    ("cluster_2_roi_20x_summary", "cluster_2_roi_20x_image_path", "_cluster_2_20x"),
]


def batch_convert_sessions(input_dir: str, output_dir: str):
    """Batch process all session directories"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create features directory
    features_dir = output_path / "features"
    features_dir.mkdir(exist_ok=True)
    
    # Collect all sessions
    sessions = sorted(input_path.glob("session_*"))
    print(f"Found {len(sessions)} session directories")
    
    samples = []
    file_mapping = {}  # Record mapping from original path to new path
    
    for session_dir in sessions:
        json_file = session_dir / "inference_results.json"
        if not json_file.exists():
            print(f"Skip {session_dir.name}: inference_results.json not found")
            continue
        
        # Read json
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        # Process each item
        for item in data:
            wsi_id = item.get("wsi_id", "unknown")
            session_name = session_dir.name
            
            for summary_key, image_key, id_suffix in FIELD_MAPPINGS:
                if summary_key in item and image_key in item:
                    old_image_path = item[image_key]
                    
                    # If it's a .pt file, move and update path
                    if old_image_path.endswith('.pt'):
                        src_file = session_dir / old_image_path
                        file_key = f"{session_name}/{old_image_path}"
                        
                        if file_key not in file_mapping:
                            if src_file.exists():
                                # New filename: session_wsi_originalfilename
                                new_filename = f"{session_name}_{wsi_id}_{Path(old_image_path).name}"
                                dst_file = features_dir / new_filename
                                shutil.copy2(src_file, dst_file)
                                file_mapping[file_key] = f"features/{new_filename}"
                            else:
                                print(f"Warning: file not found {src_file}")
                                continue
                        
                        new_image_path = file_mapping[file_key]
                    else:
                        new_image_path = old_image_path
                    
                    sample = {
                        "id": f"{session_name}_{wsi_id}{id_suffix}",
                        "image": new_image_path,
                        "conversations": [
                            {"from": "human", "value": "<image>\n"},
                            {"from": "gpt", "value": item[summary_key]}
                        ]
                    }
                    samples.append(sample)
        
        print(f"Processing complete: {session_dir.name}")
    
    # Save merged results
    output_file = output_path / "caption_merged.jsonl"
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"\nConversion complete: {len(samples)} samples -> {output_file}")
    print(f"Feature files copied to: {features_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python caption.py <input_dir> <output_dir>")
        print("Example: python caption.py /path/to/input /path/to/output")
        sys.exit(1)
    
    batch_convert_sessions(sys.argv[1], sys.argv[2])
