import os
import sys
import json
from collections import defaultdict

ROOT_DIR = sys.argv[1] if len(sys.argv) > 1 else "./output"
OUTPUT_JSON = os.path.join(ROOT_DIR, "statistics_summary.json")

image_counts = {
    "thumbnail": 0,
    "cluster": 0,
    "roi": 0
}

qa_counts = defaultdict(int)


def count_qa_block(block):
    """
    block: dict, may contain qa_single_choice / qa_multiple_choice / caption_qa
    """
    for key in ["qa_single_choice", "qa_multiple_choice", "caption_qa"]:
        if key in block and isinstance(block[key], dict):
            qa_counts[key] += len(block[key])


for root, dirs, files in os.walk(ROOT_DIR):
    # ---------- image counting ----------
    for f in files:
        if f == "thumbnail.png":
            image_counts["thumbnail"] += 1
        elif f.endswith(".png"):
            full_path = os.path.join(root, f)
            if os.sep + "clusters" + os.sep in full_path:
                image_counts["cluster"] += 1
            elif os.sep + "rois" + os.sep in full_path:
                image_counts["roi"] += 1

    # ---------- json QA counting ----------
    if "final_integrated_results.json" in files:
        json_path = os.path.join(root, "final_integrated_results.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # WSI-level
            if "wsi_results" in data:
                count_qa_block(data["wsi_results"])

            # cluster-level & roi-level
            for cluster in data.get("clusters", []):
                count_qa_block(cluster)
                for roi in cluster.get("rois", []):
                    count_qa_block(roi)

        except Exception as e:
            print(f"[Warning] Failed to read {json_path}: {e}")

# ---------- save to json ----------
summary = {
    "images": dict(image_counts),
    "qa": {
        "qa_single_choice": qa_counts["qa_single_choice"],
        "qa_multiple_choice": qa_counts["qa_multiple_choice"],
        "caption_qa": qa_counts["caption_qa"]
    }
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Statistics saved to: {OUTPUT_JSON}")
