import os
import json
import base64
import asyncio
import uuid
import mimetypes
import argparse
import shutil
import glob
import sys
from datetime import datetime
from typing import Dict, List, Any


# --- 依赖库导入 ---
import numpy as np
from PIL import Image
from openslide import OpenSlide
from openai import OpenAI

# Import prompt modules from the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 导入源代码中的提示词和异步逻辑
from prompt.wsi_bench_prompt import (
    get_vqa_prompt,
    get_infer_prompt,
    get_cluster_infer_prompt,
    get_wsi_infer_prompt,
    VQA_CATEGORIES
)

# ============================================================
# I. 配置常量 
# ============================================================
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

GEMINI_THUMBNAIL_SIZE = 2048  # WSI与Cluster用于推理的图像大小
THUMBNAIL_SIZE = 512
LAYER_PIXELS = {10: 2048, 20: 1024}
MIN_TISSUE_RATIO = 0.20       # 提取区域的最小组织占比
BACKGROUND_THRESHOLD = 220    # 用于判断背景的像素灰度阈值
MAX_CONCURRENT_REQUESTS = 9   # 最大并发 API 请求数
DEFAULT_CLUSTER_RADIUS = 1024
CLUSTER_COUNT_THRESHOLD = 100 # 忽略过小的 Cluster

# API 重试配置
MAX_RETRIES = 5  # 最大重试次数
INITIAL_RETRY_DELAY = 1  # 初始重试延迟（秒）
MAX_RETRY_DELAY = 60  # 最大重试延迟（秒）

LEVEL_TASK_MATRIX = {
    "WSI": ["1.1", "1.2", "2.1", "2.4"],
    "Cluster": ["1.2", "1.3", "2.1", "2.2"],
    "ROI": ["1.3", "1.4", "2.1", "2.2", "2.3"]
}

# ============================================================
# II. 辅助函数
# ============================================================

def ensure_dir(p: str):
    """确保目录存在，如果不存在则创建。"""
    os.makedirs(p, exist_ok=True)
    
def load_json(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj: Any, p: str):
    """保存对象为 JSON 文件，支持非 ASCII 字符。"""
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def generate_qa_id() -> str:
    """生成唯一的 QA ID。"""
    return f"qa_{uuid.uuid4().hex[:8]}"

def image_to_data_url(path: str) -> str:
    """将图像文件转换为 Base64 数据 URL，用于 API 调用。"""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{enc}"

def try_parse_json_from_text(text: str) -> Dict[str, Any]:
    """尝试从模型输出的字符串中提取 JSON 结构，处理常见的代码块封装。"""
    text = text.strip()
    try:
        return json.loads(text)
    except:
        pass
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                continue
    return {"raw_output": text, "parse_error": True}

def get_case_metadata(wsi_path: str, case_json_path: str) -> Dict:
    cases = load_json(case_json_path)
    if isinstance(cases, dict): cases = [cases]
    wsi_id = os.path.splitext(os.path.basename(wsi_path))[0]
    
    for c in cases:
        case_id = os.path.splitext(c.get("img", ""))[0]
        if case_id == wsi_id:  # 精确匹配
            return {
                "diagnosis": c.get("diagnosis", "N/A"),
                "ihc": c.get("ihc", "N/A"),
                "ptnm": c.get("ptnm", "N/A")
            }
    
    raise ValueError(f"WSI ID '{wsi_id}' not found in case info JSON.")

def get_clusters_for_svs(cluster_data: Dict, svs_basename: str) -> Dict:
    """从 Cluster Geometry 数据中提取对应 WSI 的 Cluster 信息。"""
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
    return {}

def calculate_tissue_ratio(img: Image.Image) -> float:
    """计算图像中组织（非白色背景）的占比。"""
    try:
        gray_img = img.convert('L')
        np_img = np.array(gray_img)
        tissue_pixels = np.count_nonzero(np_img < BACKGROUND_THRESHOLD)
        total_pixels = np_img.size
        return tissue_pixels / total_pixels
    except Exception as e:
        print(f"Warning: Failed to calculate tissue ratio: {e}")
        return 0.0

def cleanup_incomplete_session(output_dir, wsi_path):
    """
    清理未完成的 session 目录。
    判定为“完成”的条件：同时存在有效的
      - inference_results.json（大小 >= 100 字节）
      - vqa.jsonl（大小 >= 50 字节，至少有一行 QA）
    """
    try:
        svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
        pattern = os.path.join(output_dir, f"session_{svs_basename}_*")
        candidate_dirs = glob.glob(pattern)

        for candidate in candidate_dirs:
            if not os.path.isdir(candidate):
                continue

            json_path = os.path.join(candidate, "inference_results.json")
            jsonl_path = os.path.join(candidate, "vqa.jsonl")

            has_json = os.path.exists(json_path) and os.path.getsize(json_path) >= 100
            has_jsonl = os.path.exists(jsonl_path) and os.path.getsize(jsonl_path) >= 50

            if not (has_json and has_jsonl):
                shutil.rmtree(candidate)
                reasons = []
                if not has_json:
                    reasons.append("inference_results.json missing or too small")
                if not has_jsonl:
                    reasons.append("vqa.jsonl missing or too small")
                print(f"🗑️ Cleaned up incomplete session: {candidate} ({'; '.join(reasons)})")

    except Exception as e:
        print(f"⚠ Cleanup error: {e}")
        
def extract_all_images(wsi_path, cluster_geometry_path, output_root):
    svs_id = os.path.splitext(os.path.basename(wsi_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.abspath(os.path.join(output_root, f"session_{svs_id}_{timestamp}"))
    
    c_sub_dir = "clusters"
    r_sub_dir = "rois"
    os.makedirs(os.path.join(session_dir, c_sub_dir), exist_ok=True)
    os.makedirs(os.path.join(session_dir, r_sub_dir), exist_ok=True)

    # 复制 WSI 并生成缩略图
    shutil.copy2(wsi_path, os.path.join(session_dir, os.path.basename(wsi_path)))
    slide = OpenSlide(wsi_path)
    slide.get_thumbnail((GEMINI_THUMBNAIL_SIZE, GEMINI_THUMBNAIL_SIZE)).save(
        os.path.join(session_dir, "wsi_gemini.png")
    )
    slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE)).save(
        os.path.join(session_dir, "wsi_thumbnail.png")
    )

    with open(cluster_geometry_path, 'r') as f:
        geometry_data = json.load(f)
    clusters = geometry_data[svs_id]  # 现在和 main() 的判断一致

    # 初始化字典：只记录原始图（非缩略图、非 gemini）
    image_coords = {}   # {filename: [[x1,y1],[x2,y2]]}
    image_paths = {}    # {filename: absolute_path}

    # === 1. WSI 全图 ===
    wsi_bbox = [[0, 0], [slide.dimensions[0], slide.dimensions[1]]]
    wsi_original_name = os.path.basename(wsi_path)
    wsi_abs_path = os.path.join(session_dir, wsi_original_name)
    image_coords[wsi_original_name] = wsi_bbox
    image_paths[wsi_original_name] = wsi_abs_path

    # === 2. Cluster 原图 ===
    for c_key, c_info in clusters.items():
        if c_info.get("count", 0) <= CLUSTER_COUNT_THRESHOLD:
            continue
        cx, cy = int(c_info.get('center_x', 0)), int(c_info.get('center_y', 0))
        radius = int(float(c_info.get('radius') or DEFAULT_CLUSTER_RADIUS))
        side = radius * 2
        start_x, start_y = max(0, cx - radius), max(0, cy - radius)
        w, h = min(side, slide.dimensions[0] - start_x), min(side, slide.dimensions[1] - start_y)
        if w <= 0 or h <= 0:
            continue
        region = slide.read_region((start_x, start_y), 0, (w, h)).convert("RGB")
        m = max(w, h)
        canvas = Image.new("RGB", (m, m), (255, 255, 255))
        canvas.paste(region, ((m - w) // 2, (m - h) // 2))
        if calculate_tissue_ratio(canvas) < MIN_TISSUE_RATIO:
            continue

        # 保存 Cluster 原图（不带后缀）
        img_name = f"{c_key}.png"
        abs_path = os.path.join(session_dir, c_sub_dir, img_name)
        canvas.save(abs_path)
        cluster_bbox = [[start_x, start_y], [start_x + w, start_y + h]]
        image_coords[img_name] = cluster_bbox
        image_paths[img_name] = abs_path

        # 保存 _gemini 和 _thumbnail（不记录到 image_paths）
        c_gemini = canvas.copy()
        c_gemini.thumbnail((GEMINI_THUMBNAIL_SIZE, GEMINI_THUMBNAIL_SIZE))
        c_gemini.save(os.path.join(session_dir, c_sub_dir, f"{c_key}_gemini.png"))
        c_thumb = canvas.copy()
        c_thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        c_thumb.save(os.path.join(session_dir, c_sub_dir, f"{c_key}_thumbnail.png"))

        # === 3. ROI 原图 (10x, 20x) ===
        for mag in [10, 20]:
            target_size = LAYER_PIXELS[mag]
            rx0, ry0 = max(0, int(cx - target_size // 2)), max(0, int(cy - target_size // 2))
            rw = min(target_size, slide.dimensions[0] - rx0)
            rh = min(target_size, slide.dimensions[1] - ry0)
            if rw <= 0 or rh <= 0:
                continue
            roi_img = slide.read_region((rx0, ry0), 0, (rw, rh)).convert("RGB")
            canvas2 = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            paste_x = (target_size - rw) // 2
            paste_y = (target_size - rh) // 2
            canvas2.paste(roi_img, (paste_x, paste_y))

            roi_img_name = f"{c_key}_{mag}x.png"
            roi_abs_path = os.path.join(session_dir, r_sub_dir, roi_img_name)
            canvas2.save(roi_abs_path)
            roi_bbox = [[rx0, ry0], [rx0 + rw, ry0 + rh]]
            image_coords[roi_img_name] = roi_bbox
            image_paths[roi_img_name] = roi_abs_path

            # 保存缩略图（不记录）
            roi_thumb = canvas2.copy()
            roi_thumb.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
            roi_thumb.save(os.path.join(session_dir, r_sub_dir, f"{c_key}_{mag}x_thumbnail.png"))

    slide.close()
    return session_dir, image_coords, image_paths

# ============================================================
# III. 异步 API 任务 (ASYNC TASKS)
# ============================================================

async def gpt_call_safe(messages, client, semaphore, output_json=False, max_retries=None):
    """安全调用 LLM API，具有并发限制、错误处理和自动重试功能。
    
    Args:
        messages: API 消息列表
        client: OpenAI 客户端
        semaphore: 并发控制信号量
        output_json: 是否期望 JSON 输出
        max_retries: 最大重试次数，None 则使用全局配置 MAX_RETRIES
        
    Raises:
        RuntimeError: 当所有重试都失败时抛出异常
    """
    if max_retries is None: max_retries = MAX_RETRIES
    async with semaphore:
        last_exception = None
        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=API_MODEL, messages=messages, temperature=0.7
                )
                content = resp.choices[0].message.content
                return try_parse_json_from_text(content) if output_json else content
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"⚠ API Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ API Error: {e} (exhausted)")
        
        error_msg = str(last_exception) if last_exception else "Unknown error"
        raise RuntimeError(f"API call failed after {max_retries} retries: {error_msg}")

# --- 推理任务 ---

async def task_infer_wsi_summary(thumb_path: str, diagnosis: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    """WSI 缩略图总结推理。"""
    prompt = get_wsi_infer_prompt(diagnosis) 
    img_url = image_to_data_url(thumb_path)
    messages = [
        {"role":"system","content":"You summarize WSI macro findings."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_infer_cluster_image(cluster_id: str, cluster_img_path: str, wsi_context: str, diagnosis: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    """Cluster 图像推理。"""
    prompt = get_cluster_infer_prompt(cluster_id, wsi_context, diagnosis)
    img_url = image_to_data_url(cluster_img_path)
    messages = [
        {"role":"system","content":"You summarize cluster-level findings by integrating WSI context with the cluster image."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_infer_roi(img_path: str, mag: int, diagnosis: str, cluster_context: str, wsi_context: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    """ROI 图像推理。"""
    prompt = get_infer_prompt(mag, diagnosis, cluster_context, wsi_context)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role":"system","content":"You are a professional hepatopathologist analyzing liver biopsy images."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

# --- vqa任务 ---
async def generate_vqa_set(level_type: str, img_path: str, metadata: Dict, summary: str, mag: str, client: OpenAI, sem: asyncio.Semaphore):
    codes = LEVEL_TASK_MATRIX.get(level_type, [])
    sub_types = ["single", "multi", "caption"]
    ratio = {"single": 2, "multi": 1, "caption": 2}
    img_url = image_to_data_url(img_path)
    
    # 1️⃣ 创建所有任务
    tasks = []
    for code in codes:
        for stype in sub_types:
            for _ in range(ratio[stype]):
                prompt = get_vqa_prompt(code, stype, metadata, summary, mag)
                msgs = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]}]
                tasks.append((code, stype, msgs))

    # 2️⃣ 并发执行所有 GPT API 调用
    task_list = [gpt_call_safe(msgs, client, sem, output_json=True) for _, _, msgs in tasks]
    results = await asyncio.gather(*task_list, return_exceptions=True)

    # 3️⃣ 整理结果
    final_qas = []
    for idx, output in enumerate(results):
        code, stype, _ = tasks[idx]
        if isinstance(output, Exception):
            print(f"⚠ VQA task failed for {code}-{stype}: {output}")
            continue

        items = output if isinstance(output, list) else [output]
        for item in items:
            if isinstance(item, dict) and (item.get("question") or item.get("caption")):
                item.update({"subcategory": code, "format": stype, "uuid": str(uuid.uuid4())})
                final_qas.append(item)
    return final_qas

async def main(wsi_path: str, case_json_path: str, cluster_geometry_path: str, output_root: str):
    # 🔍 第一步：检查 cluster geometry 中是否有当前 WSI 的数据
    svs_id = os.path.splitext(os.path.basename(wsi_path))[0]
    
    if not os.path.exists(cluster_geometry_path):
        print(f"❌ Cluster geometry file not found: {cluster_geometry_path}")
        return  # 或 raise SystemExit(0) 如果你希望静默退出
    
    with open(cluster_geometry_path, 'r') as f:
        geometry_data = json.load(f)
    
    clusters = geometry_data.get(svs_id, {})
    
    if not clusters or not isinstance(clusters, dict):
        print(f"⏭️ Skipping WSI '{svs_id}': no cluster data found in {cluster_geometry_path}.")
        sys.exit(42)  # 关键：退出码 42 表示无 cluster 跳过

    # ✅ 只有存在 cluster 时，才继续
    print(f"✅ Found cluster data for WSI '{svs_id}'. Proceeding with processing...")
    
    # 1️⃣ 提取图像
    wsi_filename = os.path.basename(wsi_path)
    session_dir, image_coords, image_paths = extract_all_images(wsi_path, cluster_geometry_path, output_root)
    wsi_path_moved = os.path.join(session_dir, wsi_filename)
    gemini_path_abs = os.path.join(session_dir, "wsi_gemini.png")
    thumbnail_path_abs = os.path.join(session_dir, "wsi_thumbnail.png")

    try:
        case_metadata = get_case_metadata(wsi_path, case_json_path)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(43)  # 自定义退出码表示“缺失标注”
    diagnosis = case_metadata.get("diagnosis", "N/A")
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    def to_rel(abs_path: str) -> str:
        return os.path.relpath(abs_path, session_dir)

    # 2️⃣ 推理逻辑（保持不变）
    print("🔹 Starting WSI inference...")
    wsi_summary = await task_infer_wsi_summary(gemini_path_abs, diagnosis, client, sem)

    wsi_id = os.path.splitext(wsi_filename)[0]
    inference_results = {
        "wsi_id": wsi_id,
        "wsi_svs": wsi_filename,
        "wsi_image_path": "",
        "wsi_thumbnail_path": to_rel(thumbnail_path_abs),
        "wsi_summary": wsi_summary
    }

    clusters_dir_abs = os.path.join(session_dir, "clusters")
    cluster_files = [f for f in os.listdir(clusters_dir_abs) if f.endswith(".png") and "_thumbnail" not in f and "_gemini" not in f]

    for cluster_file in cluster_files:
        cluster_id = os.path.splitext(cluster_file)[0]
        cluster_img_path_abs = os.path.join(clusters_dir_abs, cluster_file)
        cluster_gemini_path_abs = os.path.join(clusters_dir_abs, f"{cluster_id}_gemini.png")
        cluster_thumb_path_abs = os.path.join(clusters_dir_abs, f"{cluster_id}_thumbnail.png")

        print(f"🔹 Inferring cluster {cluster_id}...")
        cluster_summary = await task_infer_cluster_image(
            cluster_id, cluster_gemini_path_abs, wsi_summary, diagnosis, client, sem
        )

        inference_results[f"{cluster_id}_id"] = cluster_id
        inference_results[f"{cluster_id}_image_path"] = to_rel(cluster_img_path_abs)
        inference_results[f"{cluster_id}_thumbnail_path"] = to_rel(cluster_thumb_path_abs)
        inference_results[f"{cluster_id}_summary"] = cluster_summary

        roi_dir_abs = os.path.join(session_dir, "rois")
        for mag in [10, 20]:
            roi_file = f"{cluster_id}_{mag}x.png"
            roi_path_abs = os.path.join(roi_dir_abs, roi_file)
            roi_thumb_path_abs = os.path.join(roi_dir_abs, f"{cluster_id}_{mag}x_thumbnail.png")

            if not os.path.exists(roi_path_abs):
                continue

            print(f"🔹 Inferring ROI {cluster_id} {mag}x...")
            roi_summary = await task_infer_roi(
                roi_path_abs, mag, diagnosis, cluster_summary, wsi_summary, client, sem
            )

            inference_results[f"{cluster_id}_roi_{mag}x_image_path"] = to_rel(roi_path_abs)
            inference_results[f"{cluster_id}_roi_{mag}x_thumbnail_path"] = to_rel(roi_thumb_path_abs)
            inference_results[f"{cluster_id}_roi_{mag}x_summary"] = roi_summary

    # 3️⃣ 保存推理结果
    json_path = os.path.join(session_dir, "inference_results.json")
    save_json(inference_results, json_path)
    print(f"✅ Inference complete. Results saved to {json_path}")

    # ============================================================
    # IV. 提取所有 .pt 特征（包括 WSI、Cluster、ROI）
    # ============================================================
    try:
        from conversation.feature import extract_region_features
    except Exception as e:
        print(f"⚠️ 无法导入 feature 模块，跳过特征提取: {e}")
    else:
        features_dir = os.path.join(session_dir, "features")
        os.makedirs(features_dir, exist_ok=True)

        # 构建 {坐标JSON字符串: .pt绝对路径}
        feature_targets = {}
        for img_name, bbox in image_coords.items():
            # 只处理原始图（已由 extract_all_images 保证）
            base_name = os.path.splitext(img_name)[0]
            pt_abs = os.path.join(features_dir, f"{base_name}.pt")
            coord_key = json.dumps(bbox)
            feature_targets[coord_key] = pt_abs

        if feature_targets:
            extract_region_features(wsi_path_moved, feature_targets)
            print(f"✅ 特征提取完成，共 {len(feature_targets)} 个区域")

            # 构建 {原图相对路径 → .pt 相对路径}
            pt_rel_map = {}
            for img_name, orig_abs in image_paths.items():
                orig_rel = to_rel(orig_abs)
                base_name = os.path.splitext(img_name)[0]
                pt_abs = os.path.join(features_dir, f"{base_name}.pt")
                pt_rel = to_rel(pt_abs)
                pt_rel_map[orig_rel] = pt_rel

            # 更新 inference_results.json
            with open(json_path, 'r', encoding='utf-8') as f:
                inf_res = json.load(f)

            updated_inf = False

            # 1. 先处理普通 image_path 字段（Cluster、ROI）
            for k, v in list(inf_res.items()):  # 使用 list() 避免迭代时修改
                if k.endswith('_image_path') and isinstance(v, str) and v in pt_rel_map:
                    inf_res[k] = pt_rel_map[v]
                    updated_inf = True

            # 2. 特别处理 WSI：它的 image_path 对应的是原始 .svs 文件的相对路径
            wsi_svs_rel = to_rel(wsi_path_moved)  # 如 "TCGA-ABC.svs"
            if wsi_svs_rel in pt_rel_map:
                inf_res["wsi_image_path"] = pt_rel_map[wsi_svs_rel]
                updated_inf = True

            if updated_inf:
                save_json(inf_res, json_path)
                print("✅ Updated image_path in inference_results.json to .pt paths")

        else:
            print("🔹 无有效区域用于特征提取。")

    # ============================================================
    # V. VQA 生成（并更新 image_path 为 .pt）
    # ============================================================
    print("🔹 Starting VQA generation...")

    qa_coroutines = []
    qa_metadatas = []

    # --- WSI VQA ---
    print("  - Scheduling WSI VQA...")
    wsi_vqa_coro = generate_vqa_set("WSI", gemini_path_abs, case_metadata, wsi_summary, "Macro", client, sem)
    qa_coroutines.append(wsi_vqa_coro)
    qa_metadatas.append({
        "level": "WSI",
        "image_path": wsi_filename,  # 原始 WSI 文件名
        "image_thumb_path": to_rel(thumbnail_path_abs),
        "belong_cluster_id": None,
        "belong_roi_mag": None
    })

    # --- Cluster & ROI VQA ---
    for cluster_file in cluster_files:
        cluster_id = os.path.splitext(cluster_file)[0]
        cluster_gemini_path_abs = os.path.join(clusters_dir_abs, f"{cluster_id}_gemini.png")
        cluster_img_path_abs = os.path.join(clusters_dir_abs, cluster_file)
        cluster_thumb_path_abs = os.path.join(clusters_dir_abs, f"{cluster_id}_thumbnail.png")
        cluster_summary = inference_results[f"{cluster_id}_summary"]

        # Cluster VQA
        print(f"  - Scheduling Cluster {cluster_id} VQA...")
        cluster_vqa_coro = generate_vqa_set("Cluster", cluster_gemini_path_abs, case_metadata, cluster_summary, "Medium", client, sem)
        qa_coroutines.append(cluster_vqa_coro)
        qa_metadatas.append({
            "level": "Cluster",
            "image_path": to_rel(cluster_img_path_abs),
            "image_thumb_path": to_rel(cluster_thumb_path_abs),
            "belong_cluster_id": cluster_id,
            "belong_roi_mag": None
        })

        # ROI VQA
        roi_dir_abs = os.path.join(session_dir, "rois")
        for mag in [10, 20]:
            roi_path_abs = os.path.join(roi_dir_abs, f"{cluster_id}_{mag}x.png")
            if not os.path.exists(roi_path_abs):
                continue
            roi_thumb_path_abs = os.path.join(roi_dir_abs, f"{cluster_id}_{mag}x_thumbnail.png")
            roi_summary = inference_results[f"{cluster_id}_roi_{mag}x_summary"]

            print(f"  - Scheduling ROI {cluster_id} {mag}x VQA...")
            roi_vqa_coro = generate_vqa_set("ROI", roi_path_abs, case_metadata, roi_summary, f"{mag}x", client, sem)
            qa_coroutines.append(roi_vqa_coro)
            qa_metadatas.append({
                "level": "ROI",
                "image_path": to_rel(roi_path_abs),
                "image_thumb_path": to_rel(roi_thumb_path_abs),
                "belong_cluster_id": cluster_id,
                "belong_roi_mag": mag
            })

    # 执行 VQA
    print(f"  - Running {len(qa_coroutines)} VQA tasks concurrently...")
    vqa_results = await asyncio.gather(*qa_coroutines, return_exceptions=True)

    # 整理结果并替换 image_path 为 .pt
    qa_entries = []
    for result, meta in zip(vqa_results, qa_metadatas):
        if isinstance(result, Exception):
            print(f"⚠ VQA task failed for {meta['level']}: {result}")
            continue

        items = result if isinstance(result, list) else [result]
        for item in items:
            if isinstance(item, dict) and (item.get("question") or item.get("caption")):
                entry = {
                    "qa_id": generate_qa_id(),
                    "image_path": meta["image_path"],
                    "image_thumb_path": meta["image_thumb_path"],
                    "belong_level": meta["level"],
                    "belong_cluster_id": meta["belong_cluster_id"],
                    "belong_roi_mag": meta["belong_roi_mag"],
                    **{k: v for k, v in item.items() if k not in ["uuid"]}
                }
                # 替换 image_path 为 .pt 路径（如果存在）
                if entry["image_path"] in pt_rel_map:
                    entry["image_path"] = pt_rel_map[entry["image_path"]]
                qa_entries.append(entry)

    # 保存 vqa.jsonl
    vqa_jsonl_path = os.path.join(session_dir, "vqa.jsonl")
    with open(vqa_jsonl_path, "w", encoding="utf-8") as f:
        for entry in qa_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ VQA generation complete. Saved {len(qa_entries)} QA pairs to {vqa_jsonl_path}")
    
# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-path", type=str, required=True)
    parser.add_argument("--case-info-json", type=str, required=True)
    parser.add_argument("--cluster-geometry", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    try:
        asyncio.run(main(args.wsi_path, args.case_info_json, args.cluster_geometry, args.output_dir))
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        cleanup_incomplete_session(args.output_dir, args.wsi_path)
        sys.exit(1)