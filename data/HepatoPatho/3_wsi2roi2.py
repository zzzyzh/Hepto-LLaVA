#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# 导入提示词模块及其内部数据和生成函数
from prompt.wsi2roi_prompts import (
    get_infer_prompt,
    get_qa_prompt,
    get_captionqa_prompt,
    get_cluster_infer_prompt,
    get_wsi_infer_prompt,
)

# ============================================================
# I. 配置常量 (CONFIGURATION)
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

# 图像处理参数
THUMBNAIL_SIZE = 2048        # WSI 缩略图输出尺寸（像素）
ROI_OUTPUT_SIZE = 512        # ROI 切片输出尺寸（像素）
CLUSTER_OUTPUT_SIZE = 2048   # Cluster 图像输出尺寸（像素）
MAGNIFICATIONS = [10, 20, 40] # 待提取的 ROI 放大倍数
LAYER_PIXELS = {10: 2048, 20: 1024, 40: 512} # 对应放大倍数下的图像区域尺寸 (Level 0 像素)

# 区域筛选和并发控制
DEFAULT_CLUSTER_RADIUS = 1024
CLUSTER_COUNT_THRESHOLD = 100 # 忽略过小的 Cluster
MIN_TISSUE_RATIO = 0.20       # 提取区域的最小组织占比
BACKGROUND_THRESHOLD = 220    # 用于判断背景的像素灰度阈值
MAX_CONCURRENT_REQUESTS = 9   # 最大并发 API 请求数
DIFFICULTIES = ['easy', 'medium', 'hard'] # QA 生成的难度级别

# API 重试配置
MAX_RETRIES = 5  # 最大重试次数
INITIAL_RETRY_DELAY = 1  # 初始重试延迟（秒）
MAX_RETRY_DELAY = 60  # 最大重试延迟（秒）


# =========================================================
# II. 辅助函数 (UTILITIES)
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

def image_to_data_url(path: str) -> str:
    """将图像文件转换为 Base64 数据 URL，用于 API 调用。"""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{enc}"

def generate_qa_id() -> str:
    """生成唯一的 QA ID。"""
    return f"qa_{uuid.uuid4().hex[:8]}"

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

def get_case_diagnosis(wsi_path: str, case_json_path: str) -> str:
    """从病例信息 JSON 文件中查找对应 WSI 的诊断结果。""" 
    cases = load_json(case_json_path)
    target = os.path.basename(wsi_path)
    svs_id = os.path.splitext(target)[0]
    if isinstance(cases, list):
        for c in cases:
            img_name = c.get("img", "")
            if img_name == target or os.path.splitext(img_name)[0] == svs_id:
                return c.get("diagnosis", "N/A")
    return "N/A"

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

def format_qa_output(qa_data: Any) -> Dict[str, Dict]:
    """格式化QA输出，将QA数据转换为带ID的字典格式"""
    if qa_data is None: return {}
    results = {}
    items = qa_data if isinstance(qa_data, list) else [qa_data]
    for item in items:
        if isinstance(item, dict) and ("question" in item or "caption" in item):
            uid = generate_qa_id()
            item['id'] = uid
            results[uid] = item
    return results

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
    """清理未完成的session目录（如果没有生成最终JSON文件）"""
    try:
        svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
        pattern = os.path.join(output_dir, f"session_{svs_basename}_*")
        incomplete_dirs = glob.glob(pattern)
        for incomplete_dir in incomplete_dirs:
            final_json = os.path.join(incomplete_dir, "final_integrated_results.json")
            if not os.path.exists(final_json):
                shutil.rmtree(incomplete_dir)
                print(f"🗑️ Cleaned up incomplete session: {incomplete_dir}")
    except Exception as e:
        print(f"⚠ Cleanup error: {e}")

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

async def task_qa_generate(img_path: str, diagnosis: str, visual_context: str, q_type: str, difficulty: str, client: OpenAI, sem: asyncio.Semaphore) -> Dict:
    """生成单选/多选 QA。"""
    prompt = get_qa_prompt(diagnosis, visual_context, q_type, difficulty)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role": "system", "content": "You are a pathology education assistant. Focus on visual evidence."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=True)

async def task_caption_qa(img_path: str, diagnosis: str, visual_context: str, mag: Any, client: OpenAI, sem: asyncio.Semaphore) -> Dict:
    """生成标题和短问答。"""
    prompt = get_captionqa_prompt(diagnosis, visual_context, mag)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role":"system","content":"You are an expert pathology education assistant."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=True)


# ============================================================
# IV. 主管道 (MAIN PIPELINE)
# ============================================================

async def main(wsi_path: str, case_info_json: str, cluster_geometry: str, output_dir: str):
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
    session_dir = os.path.join(output_dir, f"session_{svs_basename}_{timestamp}")
    roi_dir, cluster_img_dir = os.path.join(session_dir, "rois"), os.path.join(session_dir, "clusters")
    ensure_dir(session_dir); ensure_dir(roi_dir); ensure_dir(cluster_img_dir)
    
    print(f"🚀 Starting Session for: {svs_basename}")
    slide = OpenSlide(wsi_path)
    diagnosis_text = get_case_diagnosis(wsi_path, case_info_json)
    cluster_raw = load_json(cluster_geometry)
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # --- STEP 1: WSI Summary ---
    print("--- STEP 1: WSI Summary ---")
    thumb = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb_path = os.path.join(session_dir, "thumbnail.png")
    thumb.save(thumb_path)
    
    try:
        wsi_summary_text = await task_infer_wsi_summary(thumb_path, diagnosis_text, client, sem)
    except Exception as e:
        raise RuntimeError(f"WSI Summary failed: {e}")

    # --- STEP 2: Cluster Processing ---
    print("--- STEP 2: Cluster Processing ---")
    clusters = get_clusters_for_svs(cluster_raw, svs_basename)
    selected_clusters, cluster_image_map, cluster_names_ordered = [], {}, []

    for k, v in clusters.items():
        if v.get("count", 0) <= CLUSTER_COUNT_THRESHOLD: continue
        cx, cy = int(v.get('center_x', v.get('x', 0))), int(v.get('center_y', v.get('y', 0)))
        radius = int(float(v.get('radius') or v.get('r') or DEFAULT_CLUSTER_RADIUS))
        L = radius * 2
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        w, h = min(L, slide.dimensions[0] - x0), min(L, slide.dimensions[1] - y0)
        if w <= 0 or h <= 0: continue
        region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
        m = max(w, h)
        canvas = Image.new("RGB", (m, m), (255,255,255))
        canvas.paste(region, ((m - w) // 2, (m - h) // 2))
        if calculate_tissue_ratio(canvas) < MIN_TISSUE_RATIO: continue
            
        v['cluster_name'] = k
        selected_clusters.append(v)
        cluster_img_path = os.path.join(cluster_img_dir, f"{k}.png")
        canvas.resize((CLUSTER_OUTPUT_SIZE, CLUSTER_OUTPUT_SIZE), Image.LANCZOS).save(cluster_img_path)
        cluster_image_map[k] = cluster_img_path
        cluster_names_ordered.append(k)

    cluster_img_tasks = [
        asyncio.create_task(task_infer_cluster_image(c, cluster_image_map[c], wsi_summary_text, diagnosis_text, client, sem))
        for c in cluster_names_ordered
    ]
    try:
        cluster_img_sum_results = await asyncio.gather(*cluster_img_tasks)
    except Exception as e:
        for t in cluster_img_tasks: t.cancel()
        raise RuntimeError(f"Cluster inference failed: {e}")
    cluster_summary_map = dict(zip(cluster_names_ordered, cluster_img_sum_results))

    # --- STEP 3: ROI Extraction and Inference ---
    print("--- STEP 3: ROI Extraction and Inference ---")
    roi_data_map, infer_tasks = [], []
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        ctx, cx, cy = cluster_summary_map[cname], int(cluster.get('center_x', 0)), int(cluster.get('center_y', 0))
        for mag in MAGNIFICATIONS:
            size = LAYER_PIXELS.get(mag, ROI_OUTPUT_SIZE)
            rx0, ry0 = max(0, cx - size // 2), max(0, cy - size // 2)
            rw, rh = min(size, slide.dimensions[0] - rx0), min(size, slide.dimensions[1] - ry0)
            if rw <= 0 or rh <= 0: continue
            roi_img = slide.read_region((rx0, ry0), 0, (rw, rh)).convert("RGB")
            m2 = max(rw, rh)
            canvas2 = Image.new("RGB", (m2, m2), (255,255,255))
            canvas2.paste(roi_img, ((m2 - rw) // 2, (m2 - rh) // 2))
            roi_path = os.path.join(roi_dir, f"{cname}_{mag}x.png")
            canvas2.resize((ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), Image.LANCZOS).save(roi_path)
            
            t = asyncio.create_task(task_infer_roi(roi_path, mag, diagnosis_text, ctx, wsi_summary_text, client, sem))
            infer_tasks.append(t)
            roi_data_map.append({"cluster": cname, "mag": mag, "path": roi_path, "diagnosis": diagnosis_text})

    try:
        inference_results = await asyncio.gather(*infer_tasks)
    except Exception as e:
        for t in infer_tasks: t.cancel()
        raise RuntimeError(f"ROI inference failed: {e}")

    # --- STEP 4: QA Generation ---
    print("--- STEP 4: QA Generation ---")
    
    # 4.1 WSI QA
    wsi_qa_tasks = [task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "single-choice", d, client, sem) for d in DIFFICULTIES] + \
                   [task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "multiple-choice", d, client, sem) for d in DIFFICULTIES] + \
                   [task_caption_qa(thumb_path, diagnosis_text, wsi_summary_text, THUMBNAIL_SIZE, client, sem)]
    try:
        wsi_qa_res = await asyncio.gather(*wsi_qa_tasks)
    except Exception as e:
        raise RuntimeError(f"WSI QA failed: {e}")
    
    wsi_qa_s_map = dict(zip(DIFFICULTIES, wsi_qa_res[0:3]))
    wsi_qa_m_map = dict(zip(DIFFICULTIES, wsi_qa_res[3:6]))
    wsi_cap = wsi_qa_res[6]

    # 4.2 Cluster & ROI QA Helper
    async def run_qa_group(img_path, diag, ctx, mag_or_size):
        sub_tasks = []
        for d in DIFFICULTIES:
            sub_tasks.append(task_qa_generate(img_path, diag, ctx, "single-choice", d, client, sem))
            sub_tasks.append(task_qa_generate(img_path, diag, ctx, "multiple-choice", d, client, sem))
        sub_tasks.append(task_caption_qa(img_path, diag, ctx, mag_or_size, client, sem))
        res = await asyncio.gather(*sub_tasks)
        return {"s": dict(zip(DIFFICULTIES, res[0::2])), "m": dict(zip(DIFFICULTIES, res[1::2])), "c": res[-1]}

    # 4.3 Cluster QA
    c_qa_tasks = [run_qa_group(cluster_image_map[c], diagnosis_text, cluster_summary_map[c], CLUSTER_OUTPUT_SIZE) for c in cluster_names_ordered]
    try:
        c_qa_results = await asyncio.gather(*c_qa_tasks)
    except Exception as e:
        raise RuntimeError(f"Cluster QA failed: {e}")
    cluster_qa_results_map = {}
    for cname, res in zip(cluster_names_ordered, c_qa_results):
        ps, pm = {}, {}
        for d in DIFFICULTIES: ps.update(format_qa_output(res["s"][d])); pm.update(format_qa_output(res["m"][d]))
        cluster_qa_results_map[cname] = {"qa_single": ps, "qa_multi": pm, "qa_caption": format_qa_output(res["c"])}

    # 4.4 ROI QA
    r_qa_tasks = [run_qa_group(m['path'], m['diagnosis'], inference_results[i], m['mag']) for i, m in enumerate(roi_data_map)]
    try:
        r_qa_results = await asyncio.gather(*r_qa_tasks)
    except Exception as e:
        raise RuntimeError(f"ROI QA failed: {e}")
    roi_results_staging = []
    for res in r_qa_results:
        ps, pm = {}, {}
        for d in DIFFICULTIES: ps.update(format_qa_output(res["s"][d])); pm.update(format_qa_output(res["m"][d]))
        roi_results_staging.append({"qa_single": ps, "qa_multi": pm, "qa_caption": format_qa_output(res["c"])})

    # --- STEP 5: Final Assembly ---
    final_clusters_map = {}
    for i, meta in enumerate(roi_data_map):
        cname = meta['cluster']
        if cname not in final_clusters_map:
            c_qa = cluster_qa_results_map.get(cname, {})
            final_clusters_map[cname] = {
                "cluster_name": cname, "cluster_summary": cluster_summary_map.get(cname, ""),
                "cluster_image_path": cluster_image_map.get(cname, ""),
                "cluster_qa_single_choice": c_qa.get("qa_single", {}),
                "cluster_qa_multiple_choice": c_qa.get("qa_multi", {}),
                "cluster_caption_qa": c_qa.get("qa_caption", {}), "rois": []
            }
        final_clusters_map[cname]["rois"].append({
            "magnification": meta['mag'], "image_path": meta['path'], "inference_result": inference_results[i],
            "qa_single_choice": roi_results_staging[i]["qa_single"],
            "qa_multiple_choice": roi_results_staging[i]["qa_multi"], "caption_qa": roi_results_staging[i]["qa_caption"]
        })

    ws, wm = {}, {}
    for d in DIFFICULTIES: ws.update(format_qa_output(wsi_qa_s_map[d])); wm.update(format_qa_output(wsi_qa_m_map[d]))
    final_output = {
        "session_info": {"timestamp": timestamp, "wsi_path": wsi_path, "diagnosis": diagnosis_text},
        "wsi_results": {
            "thumbnail_path": thumb_path, "inference_summary": wsi_summary_text,
            "qa_single_choice": ws, "qa_multiple_choice": wm, "caption_qa": format_qa_output(wsi_cap)
        },
        "clusters": list(final_clusters_map.values())
    }
    save_json(final_output, os.path.join(session_dir, "final_integrated_results.json"))
    print(f"✅ Pipeline Finished!")

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