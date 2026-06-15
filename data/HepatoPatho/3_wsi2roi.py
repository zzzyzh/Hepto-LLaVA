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


# ============================================================
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
    
    # 尝试解析 Markdown 代码块
    if "```" in text:
        import re
        match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
    
    # 尝试解析最外层的 {} 或 [] 结构
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
            # 匹配完整文件名或文件名ID
            if img_name == target or os.path.splitext(img_name)[0] == svs_id:
                return c.get("diagnosis", "N/A")
    return "N/A"

def get_clusters_for_svs(cluster_data: Dict, svs_basename: str) -> Dict:
    """从 Cluster Geometry 数据中提取对应 WSI 的 Cluster 信息。"""
    clusters_found = cluster_data.get(svs_basename, {})
    
    # 如果找到的是嵌套结构（兼容不同格式），则进行扁平化处理
    flat_clusters = {}
    if isinstance(clusters_found, dict):
        for k, v in clusters_found.items():
            if isinstance(v, dict) and ("count" in v or "center_x" in v):
                flat_clusters[k] = v
            elif isinstance(v, dict):
                # 兼容 { 'level': { 'cluster_id': {data} } } 的格式
                for kk, vv in v.items():
                     if isinstance(vv, dict) and ("count" in vv or "center_x" in vv):
                         flat_clusters[kk] = vv
        return flat_clusters if flat_clusters else clusters_found
    return {}

def format_qa_output(qa_data: Any) -> Dict[str, Dict]:
    """格式化QA输出，将QA数据转换为带ID的字典格式"""
    if qa_data is None:
        return {}
        
    results = {}
    items = qa_data if isinstance(qa_data, list) else [qa_data]
    for item in items:
        # 仅处理包含关键 QA 字段的字典
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
        # 使用配置的阈值进行筛选
        tissue_pixels = np.count_nonzero(np_img < BACKGROUND_THRESHOLD)
        total_pixels = np_img.size
        return tissue_pixels / total_pixels
    except Exception as e:
        print(f"Warning: Failed to calculate tissue ratio: {e}")
        return 0.0

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
    if max_retries is None:
        max_retries = MAX_RETRIES

    async with semaphore:
        last_exception = None
        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=API_MODEL,
                    messages=messages,
                    temperature=0.7
                )
                content = resp.choices[0].message.content
                if output_json:
                    return try_parse_json_from_text(content)
                return content
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # 指数退避：延迟时间 = min(INITIAL_RETRY_DELAY * (2^attempt), MAX_RETRY_DELAY)
                    wait_time = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    print(f"⚠ API Error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"   Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"❌ API Error: {e} (all {max_retries} retries exhausted)")

        # 所有重试都失败，抛出异常终止处理
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

# --- QA 任务 ---

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

# main 函数明确接受四个关键字参数
async def main(wsi_path: str, case_info_json: str, cluster_geometry: str, output_dir: str):
    
    # 1. 初始化和环境准备
    WSI_PATH = wsi_path
    CASE_INFO_JSON = case_info_json
    CLUSTER_GEOMETRY = cluster_geometry
    OUTPUT_DIR = output_dir

    ensure_dir(OUTPUT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svs_basename = os.path.splitext(os.path.basename(WSI_PATH))[0]
    session_dir = os.path.join(OUTPUT_DIR, f"session_{svs_basename}_{timestamp}")
    roi_dir = os.path.join(session_dir, "rois")
    cluster_img_dir = os.path.join(session_dir, "clusters")
    ensure_dir(session_dir)
    ensure_dir(roi_dir)
    ensure_dir(cluster_img_dir)
    
    print(f"🚀 Starting Session for: {svs_basename} at {session_dir}")
    
    if not os.path.exists(WSI_PATH):
        print(f"❌ Error: WSI file not found at {WSI_PATH}")
        return

    slide = OpenSlide(WSI_PATH)
    diagnosis_text = get_case_diagnosis(WSI_PATH, CASE_INFO_JSON)
    cluster_raw = load_json(CLUSTER_GEOMETRY)
    
    print(f"ℹ Diagnosis Context: {diagnosis_text}")

    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    
    # 2. WSI 缩略图提取和全局总结
    print("--- STEP 1: WSI Summary ---")
    thumb = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb_path = os.path.join(session_dir, "thumbnail.png")
    thumb.save(thumb_path)
    
    wsi_summary_text = await task_infer_wsi_summary(thumb_path, diagnosis_text, client, sem)
    print("✔ WSI Summary Done.")
    
    
    # 3. Cluster 提取、筛选和总结
    print("--- STEP 2: Cluster Processing ---")
    clusters = get_clusters_for_svs(cluster_raw, svs_basename)
    selected_clusters = []
    cluster_image_map = {}
    cluster_names_ordered = []

    # 3.1 Cluster 图像提取和筛选
    for k, v in clusters.items():
        if v.get("count", 0) <= CLUSTER_COUNT_THRESHOLD: continue
            
        cname = k
        cx = int(v.get('center_x', v.get('x', 0)))
        cy = int(v.get('center_y', v.get('y', 0)))
        
        radius = int(float(v.get('radius') or v.get('r') or DEFAULT_CLUSTER_RADIUS))
        L = radius * 2
        x0, y0 = max(0, cx - radius), max(0, cy - radius)
        w, h = min(L, slide.dimensions[0] - x0), min(L, slide.dimensions[1] - y0)

        if w <= 0 or h <= 0: continue

        region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
        
        # 填充并筛选
        m = max(w, h)
        canvas = Image.new("RGB", (m, m), (255,255,255))
        canvas.paste(region, ((m - w) // 2, (m - h) // 2))
        region_sq = canvas
            
        if calculate_tissue_ratio(region_sq) < MIN_TISSUE_RATIO:
            print(f"⚠ Skipping cluster {cname}: Tissue ratio too low.")
            continue
            
        v['cluster_name'] = cname
        selected_clusters.append(v)
        
        cluster_img = region_sq.resize((CLUSTER_OUTPUT_SIZE, CLUSTER_OUTPUT_SIZE), Image.LANCZOS)
        cluster_img_path = os.path.join(cluster_img_dir, f"{cname}.png")
        cluster_img.save(cluster_img_path)
        cluster_image_map[cname] = cluster_img_path
        cluster_names_ordered.append(cname)

    print(f"ℹ Selected {len(selected_clusters)} clusters for analysis.")

    # 3.2 Cluster 总结推理
    cluster_img_tasks = [
        asyncio.create_task(task_infer_cluster_image(cname, cluster_image_map[cname], wsi_summary_text, diagnosis_text, client, sem))
        for cname in cluster_names_ordered
    ]
    cluster_img_sum_results = await asyncio.gather(*cluster_img_tasks)
    cluster_summary_map = dict(zip(cluster_names_ordered, cluster_img_sum_results))
    print("✔ Cluster Summaries Inferred.")

    
    # 4. ROI 提取和推理
    print("--- STEP 3: ROI Extraction and Inference ---")
    roi_data_map = []
    infer_tasks = []
    
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        if cname not in cluster_summary_map: continue
            
        cluster_context = cluster_summary_map[cname]
        cx = int(cluster.get('center_x', 0))
        cy = int(cluster.get('center_y', 0))

        for mag in MAGNIFICATIONS:
            # ROI 区域计算和提取
            size = LAYER_PIXELS.get(mag, ROI_OUTPUT_SIZE)
            rx0, ry0 = max(0, cx - size // 2), max(0, cy - size // 2)
            rx_w, rx_h = min(size, slide.dimensions[0] - rx0), min(size, slide.dimensions[1] - ry0)
            if rx_w <= 0 or rx_h <= 0: continue
            
            region_roi = slide.read_region((rx0, ry0), 0, (rx_w, rx_h)).convert("RGB")
            
            m2 = max(rx_w, rx_h)
            canvas2 = Image.new("RGB", (m2, m2), (255,255,255))
            canvas2.paste(region_roi, ((m2 - rx_w) // 2, (m2 - rx_h) // 2))
            roi_img = canvas2.resize((ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), Image.LANCZOS)
            
            roi_path = os.path.join(roi_dir, f"{cname}_{mag}x.png")
            roi_img.save(roi_path)

            t_infer = asyncio.create_task(
                task_infer_roi(roi_path, mag, diagnosis_text, cluster_context, wsi_summary_text, client, sem)
            )
            infer_tasks.append(t_infer)
            roi_data_map.append({
                "cluster": cname, "mag": mag, "path": roi_path, "diagnosis": diagnosis_text, "cluster_context": cluster_context
            })

    inference_results = await asyncio.gather(*infer_tasks)
    print("✔ ROI Inferences Completed.")

    
    # 5. QA 生成
    print("--- STEP 4: QA Generation ---")
    
    # WSI QA 任务
    wsi_qa_s_tasks = [task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "single-choice", d, client, sem) for d in DIFFICULTIES]
    wsi_qa_m_tasks = [task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "multiple-choice", d, client, sem) for d in DIFFICULTIES]
    wsi_cap_task = task_caption_qa(thumb_path, diagnosis_text, wsi_summary_text, THUMBNAIL_SIZE, client, sem)
    
    wsi_qa_s_map = dict(zip(DIFFICULTIES, await asyncio.gather(*wsi_qa_s_tasks)))
    wsi_qa_m_map = dict(zip(DIFFICULTIES, await asyncio.gather(*wsi_qa_m_tasks)))
    wsi_cap = await wsi_cap_task
    
    # Cluster QA 任务
    cluster_qa_tasks = []
    for cname in cluster_names_ordered:
        summary = cluster_summary_map.get(cname, "")
        cluster_img_path = cluster_image_map.get(cname)
        if cluster_img_path and summary:
            async def run_cluster_qa_group(cname, img_path, diag, ctx):
                qa_singles, qa_multis = {}, {}
                for d in DIFFICULTIES:
                    t_s = await task_qa_generate(img_path, diag, ctx, "single-choice", d, client, sem)
                    t_m = await task_qa_generate(img_path, diag, ctx, "multiple-choice", d, client, sem)
                    qa_singles[d] = t_s
                    qa_multis[d] = t_m
                t_c = await task_caption_qa(img_path, diag, ctx, CLUSTER_OUTPUT_SIZE, client, sem)
                return cname, qa_singles, qa_multis, t_c
            cluster_qa_tasks.append(run_cluster_qa_group(cname, cluster_img_path, diagnosis_text, summary))
            
    cluster_qa_results_list = await asyncio.gather(*cluster_qa_tasks)
    cluster_qa_results_map = {}
    for cname, res_s_map, res_m_map, res_c in cluster_qa_results_list:
        processed_singles, processed_multis = {}, {}
        for d in DIFFICULTIES: processed_singles.update(format_qa_output(res_s_map[d]))
        for d in DIFFICULTIES: processed_multis.update(format_qa_output(res_m_map[d]))
        cluster_qa_results_map[cname] = {
            "qa_single": processed_singles,
            "qa_multi": processed_multis,
            "qa_caption": format_qa_output(res_c)
        }
    
    # ROI QA 任务
    qa_tasks = []
    for i, meta in enumerate(roi_data_map):
        inf_res = inference_results[i]
        async def run_roi_qa_group(idx, m_path, m_diag, m_ctx, m_mag):
            qa_singles, qa_multis = {}, {}
            for d in DIFFICULTIES:
                t_s = await task_qa_generate(m_path, m_diag, m_ctx, "single-choice", d, client, sem)
                t_m = await task_qa_generate(m_path, m_diag, m_ctx, "multiple-choice", d, client, sem)
                qa_singles[d] = t_s
                qa_multis[d] = t_m
            t_c = await task_caption_qa(m_path, m_diag, m_ctx, m_mag, client, sem)
            return idx, qa_singles, qa_multis, t_c
        qa_tasks.append(run_roi_qa_group(i, meta['path'], meta['diagnosis'], inf_res, meta['mag']))

    roi_qa_results_list = await asyncio.gather(*qa_tasks)
    roi_results_staging = [None] * len(roi_data_map)
    for idx, res_s_map, res_m_map, res_c in roi_qa_results_list:
        processed_singles, processed_multis = {}, {}
        for d in DIFFICULTIES: processed_singles.update(format_qa_output(res_s_map[d]))
        for d in DIFFICULTIES: processed_multis.update(format_qa_output(res_m_map[d]))
        roi_results_staging[idx] = {
            "qa_single": processed_singles,
            "qa_multi": processed_multis,
            "qa_caption": format_qa_output(res_c)
        }
    
    print("✔ All QA Tasks Completed.")

    
    # 6. 结果组装和保存
    print("--- STEP 5: Final Assembly and Save ---")
    
    final_clusters_map = {}
    for i, meta in enumerate(roi_data_map):
        cname = meta['cluster']
        
        if cname not in final_clusters_map:
            cluster_qa = cluster_qa_results_map.get(cname, {})
            final_clusters_map[cname] = {
                "cluster_name": cname,
                "cluster_summary": cluster_summary_map.get(cname, ""),
                "cluster_image_path": cluster_image_map.get(cname, ""),
                "cluster_qa_single_choice": cluster_qa.get("qa_single", {}),
                "cluster_qa_multiple_choice": cluster_qa.get("qa_multi", {}),
                "cluster_caption_qa": cluster_qa.get("qa_caption", {}),
                "rois": []
            }
        
        # 将 ROI 推理结果和 QA 结果合并到其对应的 Cluster 下
        final_clusters_map[cname]["rois"].append({
            "magnification": meta['mag'],
            "image_path": meta['path'],
            "inference_result": inference_results[i],
            "qa_single_choice": roi_results_staging[i]["qa_single"],
            "qa_multiple_choice": roi_results_staging[i]["qa_multi"],
            "caption_qa": roi_results_staging[i]["qa_caption"]
        })

    # WSI QA 结果处理
    processed_wsi_s, processed_wsi_m = {}, {}
    for d in DIFFICULTIES:
        processed_wsi_s.update(format_qa_output(wsi_qa_s_map[d]))
        processed_wsi_m.update(format_qa_output(wsi_qa_m_map[d]))

    final_output = {
        "session_info": {
            "timestamp": timestamp,
            "wsi_path": WSI_PATH,
            "diagnosis": diagnosis_text
        },
        "wsi_results": {
            "thumbnail_path": thumb_path,
            "inference_summary": wsi_summary_text,
            "qa_single_choice": processed_wsi_s,
            "qa_multiple_choice": processed_wsi_m,
            "caption_qa": format_qa_output(wsi_cap)
        },
        "clusters": list(final_clusters_map.values())
    }

    out_json_path = os.path.join(session_dir, "final_integrated_results.json")
    save_json(final_output, out_json_path)

    print(f"\n✅ Pipeline Finished! Results saved to:\n{out_json_path}")


# ============================================================
# V. 脚本入口 (ENTRY POINT)
# ============================================================

def parse_args():
    """解析命令行参数，只包含必需的四个路径。"""
    parser = argparse.ArgumentParser(description="处理WSI图像并生成诊断报告和QA")
    # 参数名使用短横线 `-`，在 args 对象中访问时为下划线 `_`
    parser.add_argument("--wsi-path", type=str, required=True, help="WSI文件路径")
    parser.add_argument("--case-info-json", type=str, required=True, help="病例信息JSON文件路径")
    parser.add_argument("--cluster-geometry", type=str, required=True, help="Cluster几何信息JSON文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        asyncio.run(main(
            wsi_path=args.wsi_path,
            case_info_json=args.case_info_json,
            cluster_geometry=args.cluster_geometry,
            output_dir=args.output_dir
        ))
    except Exception as e:
        print(f"❌ An unexpected error occurred during pipeline execution: {e}")
        exit(1)