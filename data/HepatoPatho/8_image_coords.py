import os
import json
import argparse
from openslide import OpenSlide
from typing import Dict, Any

# 配置
DEFAULT_CLUSTER_RADIUS = 256
CLUSTER_COUNT_THRESHOLD = 0
LAYER_PIXELS = {10: 2048, 20: 1024}


def get_clusters_for_svs(cluster_data: Dict, svs_basename: str) -> Dict:
    clusters_found = cluster_data.get(svs_basename, {})
    flat_clusters = {}
    if isinstance(clusters_found, dict):
        for k, v in clusters_found.items():
            if isinstance(v, dict) and ("count" in v or "center_x" in v):
                flat_clusters[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and ("count" in vv or "center_x" in vv):
                        flat_clusters[kk] = vv
    return flat_clusters if flat_clusters else clusters_found


def load_wsi_id_list(txt_path: str):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def main(wsi_dir: str, wsi_id_txt: str, cluster_geometry_path: str, output_json: str):
    wsi_ids = load_wsi_id_list(wsi_id_txt)
    print(f"📋 Loaded {len(wsi_ids)} WSI IDs")

    with open(cluster_geometry_path, 'r') as f:
        cluster_data = json.load(f)

    flat_output: Dict[str, Dict] = {}

    for wsi_id in wsi_ids:
        svs_path = os.path.join(wsi_dir, f"{wsi_id}.svs")
        if not os.path.exists(svs_path):
            print(f"⚠️  Skipped (not found): {wsi_id}")
            continue

        try:
            slide = OpenSlide(svs_path)
            w, h = slide.dimensions
            slide.close()

            # Add WSI entry
            wsi_key = f"{wsi_id}.png"
            flat_output[wsi_key] = {
                "type": "wsi",
                "wsi_id": wsi_id,
                "bbox": [[0, 0], [w, h]]
            }

            # Process clusters
            clusters = get_clusters_for_svs(cluster_data, wsi_id)
            if not clusters:
                print(f"⏭️  No clusters for {wsi_id}")
                continue

            for c_key, c_info in clusters.items():
                count = c_info.get("count", 0)
                if count <= CLUSTER_COUNT_THRESHOLD:
                    continue

                cx = int(c_info.get("center_x", 0))
                cy = int(c_info.get("center_y", 0))
                radius = int(float(c_info.get("radius") or DEFAULT_CLUSTER_RADIUS))

                x0 = max(0, cx - radius)
                y0 = max(0, cy - radius)
                x1 = min(w, cx + radius)
                y1 = min(h, cy + radius)
                if x1 <= x0 or y1 <= y0:
                    continue

                # Cluster entry
                cluster_key = f"{wsi_id}_{c_key}.png"
                flat_output[cluster_key] = {
                    "type": "cluster",
                    "wsi_id": wsi_id,
                    "cluster_id": c_key,
                    "bbox": [[x0, y0], [x1, y1]]
                }

                # ROI entries (10x, 20x)
                for mag in [10, 20]:
                    size = LAYER_PIXELS[mag]
                    rx0 = max(0, cx - size // 2)
                    ry0 = max(0, cy - size // 2)
                    rx1 = min(w, rx0 + size)
                    ry1 = min(h, ry0 + size)
                    if rx1 <= rx0 or ry1 <= ry0:
                        continue

                    roi_key = f"{wsi_id}_{c_key}_{mag}x.png"
                    flat_output[roi_key] = {
                        "type": "roi",
                        "wsi_id": wsi_id,
                        "cluster_id": c_key,
                        "magnification": f"{mag}x",
                        "bbox": [[rx0, ry0], [rx1, ry1]]
                    }

            print(f"✅ Processed: {wsi_id}")

        except Exception as e:
            print(f"❌ Error on {wsi_id}: {e}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(flat_output, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done! Saved {len(flat_output)} entries to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate flat coordinate JSON with .csv-style IDs like '3-2-C-6_cluster_2_10x.csv'"
    )
    parser.add_argument("--wsi-dir", type=str, required=True, help="Directory of .svs files")
    parser.add_argument("--wsi-id-txt", type=str, required=True, help="TXT file with WSI IDs (one per line)")
    parser.add_argument("--cluster-geometry", type=str, required=True, help="Cluster geometry JSON")
    parser.add_argument("--output-json", type=str, required=True, help="Output JSON path")

    args = parser.parse_args()
    main(args.wsi_dir, args.wsi_id_txt, args.cluster_geometry, args.output_json)