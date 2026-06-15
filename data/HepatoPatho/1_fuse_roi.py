import os
import numpy as np
import argparse
import h5py
import warnings
import sys
import heapq
import json
from joblib import Parallel, delayed
from sklearn.neighbors import KDTree
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw
from scipy import ndimage

def load_slide_h5_data(h5_path, feat_key='features'):
    try:
        with h5py.File(h5_path, 'r') as hf:
            if feat_key not in hf:
                return None
            data = np.array(hf[feat_key])
            return data
    except Exception:
        return None

def load_patch_h5_data(h5_path, feat_key='features', coords_key='coords'):
    try:
        with h5py.File(h5_path, 'r') as hf:
            if feat_key not in hf or coords_key not in hf:
                return None, None
            features = np.array(hf[feat_key])
            coords = np.array(hf[coords_key]) # (x, y)
            return features, coords
    except Exception:
        return None, None
        
def get_slide_ids_from_dir(directory):
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.h5')]
        ids = set([os.path.splitext(f)[0] for f in files])
        return ids
    except Exception as e:
        print(f"Error scanning dir: {e}")
        return None

def find_closest_patch_idx(all_coords, target_coord):
    distances = np.sum((all_coords - target_coord)**2, axis=1)
    return np.argmin(distances)

def get_largest_inscribed_circle(grid_mask):
    if not np.any(grid_mask):
        return 0, 0, 0
    dist_map = ndimage.distance_transform_edt(grid_mask)
    max_dist = np.max(dist_map)
    max_loc = np.unravel_index(np.argmax(dist_map), dist_map.shape) # (y, x)
    return max_dist, max_loc[0], max_loc[1]

def create_cluster_visualization(
    patch_coords: np.ndarray, 
    patch_labels: np.ndarray, 
    patch_size: float, 
    save_path: str,
    circles_info: list = None,
    scale_factor: int = 10
):
    COLOR_MAP = [
        [255, 255, 255], # 0: Background
        [255, 150, 150], # 1: Cluster 0
        [150, 255, 150], # 2: Cluster 1
        [150, 150, 255], # 3: Cluster 2
    ]
    colors = np.array(COLOR_MAP, dtype=np.uint8)
    
    min_coords = np.min(patch_coords, axis=0)
    coords_norm = patch_coords - min_coords
    grid_coords = np.round(coords_norm / patch_size).astype(int)
    
    max_x = np.max(grid_coords[:, 0])
    max_y = np.max(grid_coords[:, 1])
    canvas_shape = (max_y + 2, max_x + 2, 3)
    
    canvas = np.zeros(canvas_shape, dtype=np.uint8)
    canvas[:, :] = colors[0] 
    
    for i in range(len(grid_coords)):
        x_g, y_g = grid_coords[i]
        label_with_offset = patch_labels[i] + 1
        if label_with_offset < len(colors):
            canvas[y_g, x_g] = colors[label_with_offset]
        
    img = Image.fromarray(canvas)
    
    if circles_info:
        draw = ImageDraw.Draw(img)
        for (cy, cx, r) in circles_info:
            left_up = (cx - r, cy - r)
            right_down = (cx + r, cy + r)
            draw.ellipse([left_up, right_down], outline=(0, 0, 0), width=1)
            draw.point((cx, cy), fill=(0, 0, 0))

    new_size = (canvas_shape[1] * scale_factor, canvas_shape[0] * scale_factor)
    img = img.resize(new_size, Image.Resampling.NEAREST)
    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # img = img.transpose(Image.ROTATE_270)
    img.save(save_path)

def process_slide(slide_id, args):
    K_CLUSTERS = 3
    
    slide_feat_path = os.path.join(args.slide_feat_dir, f"{slide_id}.h5")
    patch_feat_path = os.path.join(args.patch_feat_dir, f"{slide_id}.h5")
    
    # 1. Load Data
    try:
        slide_feat = load_slide_h5_data(slide_feat_path, args.feat_key)
        patch_feats, patch_coords = load_patch_h5_data(patch_feat_path, args.feat_key, args.coords_key)
        
        if slide_feat is None or patch_feats is None or patch_coords is None:
            return None
        if slide_feat.ndim > 1: slide_feat = slide_feat.squeeze()
    except Exception as e:
        print(f"[{slide_id}] Load Error: {e}", flush=True)
        return None
        
    N_patches = patch_feats.shape[0]
    if N_patches <= K_CLUSTERS:
        return None
    
    # 2. Init Graph
    try:
        kdt = KDTree(patch_coords)
        distances, _ = kdt.query(patch_coords, k=2)
        min_dist = np.min(distances[:, 1])
        patch_size = min_dist if min_dist > 1e-6 else 512.0
        radius = 1.42 * patch_size
        adj_list_with_self = kdt.query_radius(patch_coords, r=radius, return_distance=False)
        adj_map = {i: set(adj_list_with_self[i]) - {i} for i in range(N_patches)}
    except Exception as e:
        print(f"[{slide_id}] KDTree Error: {e}", flush=True)
        return None

    # 3. Seeds (Triangle)
    min_coords = np.min(patch_coords, axis=0)
    max_coords = np.max(patch_coords, axis=0)
    center_coords = (min_coords + max_coords) / 2
    dims = max_coords - min_coords
    width, height = dims[0], dims[1]
    
    cx, cy = center_coords[0], center_coords[1]
    grid_centers = [
        [cx, cy + height * 0.25],           
        [cx - width * 0.25, cy - height * 0.25],
        [cx + width * 0.25, cy - height * 0.25],
    ]
    seed_indices = [find_closest_patch_idx(patch_coords, center) for center in grid_centers]
    
    # 4. Region Growing
    patch_labels = np.full(N_patches, -1, dtype=int)
    cluster_means = patch_feats[seed_indices, :].astype(np.float64)
    pq = []
    
    for k in range(K_CLUSTERS):
        patch_labels[seed_indices[k]] = k

    unassigned_set = set(np.where(patch_labels == -1)[0])
    initial_frontier = set().union(*(adj_map[idx] for idx in seed_indices)).intersection(unassigned_set)

    for patch_idx in initial_frontier:
        patch_feat = patch_feats[patch_idx].reshape(1, -1)
        best_k = -1
        best_sim = -np.inf
        for k in range(K_CLUSTERS):
            if adj_map[patch_idx].intersection({seed_indices[k]}):
                sim = cosine_similarity(patch_feat, cluster_means[k].reshape(1, -1))[0, 0]
                if sim > best_sim:
                    best_sim = sim
                    best_k = k
        if best_k != -1:
            heapq.heappush(pq, (-best_sim, patch_idx, best_k))

    cluster_n = [1] * K_CLUSTERS
    
    while pq and unassigned_set:
        _, patch_idx, cluster_k = heapq.heappop(pq)
        if patch_labels[patch_idx] != -1: continue
        patch_labels[patch_idx] = cluster_k
        unassigned_set.remove(patch_idx)
        cluster_n[cluster_k] += 1
        n = cluster_n[cluster_k]
        cluster_means[cluster_k] = ((cluster_means[cluster_k] * (n - 1)) + patch_feats[patch_idx]) / n
        
        new_neighbors = adj_map[patch_idx].intersection(unassigned_set)
        for new_neighbor_idx in new_neighbors:
            new_patch_feat = patch_feats[new_neighbor_idx].reshape(1, -1)
            best_k_new = -1
            best_sim_new = -np.inf
            for k_all in range(K_CLUSTERS):
                if adj_map[new_neighbor_idx].intersection(np.where(patch_labels == k_all)[0]):
                    sim = cosine_similarity(new_patch_feat, cluster_means[k_all].reshape(1, -1))[0, 0]
                    if sim > best_sim_new:
                        best_sim_new = sim
                        best_k_new = k_all
            if best_k_new != -1:
                heapq.heappush(pq, (-best_sim_new, new_neighbor_idx, best_k_new))

    # Fallback
    fallback_indices = np.where(patch_labels == -1)[0]
    if len(fallback_indices) > 0:
        sims = cosine_similarity(patch_feats[fallback_indices], cluster_means)
        best_ks = np.argmax(sims, axis=1)
        patch_labels[fallback_indices] = best_ks

    # --- 5. Calculate Geometry & Sizes ---
    coords_norm = patch_coords - min_coords
    grid_coords = np.round(coords_norm / patch_size).astype(int)
    
    max_x_g = np.max(grid_coords[:, 0])
    max_y_g = np.max(grid_coords[:, 1])
    grid_shape = (max_y_g + 1, max_x_g + 1)
    
    slide_geometry_info = {}
    circles_viz_data = []
    cluster_sizes_arr = np.zeros(K_CLUSTERS, dtype=int) # *** 新增: 用于NPZ保存 ***
    
    for k in range(K_CLUSTERS):
        mask = np.zeros(grid_shape, dtype=bool)
        cluster_indices = np.where(patch_labels == k)[0]
        count = len(cluster_indices) # *** 新增: 计算数量 ***
        cluster_sizes_arr[k] = count
        
        if count == 0:
            slide_geometry_info[f"cluster_{k}"] = {
                "center_x": 0, "center_y": 0, "radius": 0, "count": 0
            }
            continue
            
        pts = grid_coords[cluster_indices]
        mask[pts[:, 1], pts[:, 0]] = True
        
        r_grid, cy_grid, cx_grid = get_largest_inscribed_circle(mask)
        
        center_orig_x = float(cx_grid * patch_size + min_coords[0])
        center_orig_y = float(cy_grid * patch_size + min_coords[1])
        radius_orig = float(r_grid * patch_size)
        
        slide_geometry_info[f"cluster_{k}"] = {
            "center_x": center_orig_x,
            "center_y": center_orig_y,
            "radius": radius_orig,
            "count": int(count) # *** 新增: 保存到 JSON ***
        }
        circles_viz_data.append((cy_grid, cx_grid, r_grid))

    # --- 6. Visualize ---
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, "cluster_visuals")
        os.makedirs(viz_dir, exist_ok=True)
        save_path = os.path.join(viz_dir, f"{slide_id}_clusters.png")
        create_cluster_visualization(
            patch_coords, patch_labels, patch_size, save_path, 
            circles_info=circles_viz_data
        )

    # --- 7. Save Features ---
    virtual_features = np.zeros((K_CLUSTERS, patch_feats.shape[1]))
    for k in range(K_CLUSTERS):
        members = np.where(patch_labels == k)[0]
        if len(members) > 0:
            virtual_features[k] = np.mean(patch_feats[members], axis=0)
            
    np.savez(
        os.path.join(args.output_dir, f"{slide_id}_fused_features.npz"),
        slide_feat=slide_feat,
        virtual_patch_feats=virtual_features,
        cluster_sizes=cluster_sizes_arr # *** 新增: 保存数量到 NPZ ***
    )
    
    return {
        "slide_id": slide_id,
        "geometry": slide_geometry_info
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_feat_dir', type=str, required=True)
    parser.add_argument('--patch_feat_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--feat_key', type=str, default='features')
    parser.add_argument('--coords_key', type=str, default='coords')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    slide_ids = get_slide_ids_from_dir(args.slide_feat_dir)
    patch_ids = get_slide_ids_from_dir(args.patch_feat_dir)
    if not slide_ids or not patch_ids:
        print("Input directory error.")
        return
        
    common_ids = sorted(list(slide_ids.intersection(patch_ids)))
    
    print(f"Processing {len(common_ids)} slides (K=3)...")
    
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_slide)(sid, args) for sid in common_ids
    )
    
    final_json_data = {}
    for res in results:
        if res is not None:
            final_json_data[res['slide_id']] = res['geometry']
            
    json_path = os.path.join(args.output_dir, "cluster_geometry.json")
    with open(json_path, 'w') as f:
        json.dump(final_json_data, f, indent=4)
        
    print(f"Geometry JSON saved to: {json_path}")

if __name__ == "__main__":
    main()