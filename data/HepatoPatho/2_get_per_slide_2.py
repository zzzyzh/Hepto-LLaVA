#相较2增加组织占比判断，避免对大片空白cluster进行推理
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

# 注意：为了运行，您需要安装这些库：
# pip install numpy pillow openslide-python openai
import numpy as np
from PIL import Image
from openslide import OpenSlide
from openai import OpenAI

# 导入提示词生成模块
from prompt.prompt import (
    get_infer_prompt, 
    get_qa_prompt, 
    get_captionqa_prompt,
    get_cluster_infer_prompt,
    get_wsi_summary_prompt
)

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

# 图像参数
THUMBNAIL_SIZE = 2048
ROI_OUTPUT_SIZE = 512
# cluster 输出图像尺寸
CLUSTER_OUTPUT_SIZE = 2048

# magnification 相关参数
MAGNIFICATIONS = [10, 20, 40]
LAYER_PIXELS = {10: 2048, 20: 1024, 40: 512}

# 筛选参数
CLUSTER_COUNT_THRESHOLD = 100


MIN_TISSUE_RATIO = 0.20  # 至少需要 20% 的区域是组织
BACKGROUND_THRESHOLD = 220  # 灰度值大于 220 的像素被视为背景

# 并发控制
MAX_CONCURRENT_REQUESTS = 15

# API 重试配置
MAX_RETRIES = 5  # 最大重试次数
INITIAL_RETRY_DELAY = 1  # 初始重试延迟（秒）
MAX_RETRY_DELAY = 60  # 最大重试延迟（秒）

# QA 难度等级
DIFFICULTIES = ['easy', 'medium', 'hard']


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def image_to_data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{enc}"

def generate_qa_id():
    return f"qa_{uuid.uuid4().hex[:8]}"

def try_parse_json_from_text(text):
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

def get_case_diagnosis(wsi_path, case_json_path):
    cases = load_json(case_json_path)
    if not isinstance(cases, list):
        return "N/A"
    
    target = os.path.basename(wsi_path)
    svs_id = os.path.splitext(target)[0]
    
    for c in cases:
        img = c.get("img", "")
        if img == target or os.path.splitext(img)[0] == svs_id:
            return c.get("diagnosis", "N/A")
    return "N/A"

def get_clusters_for_svs(cluster_data, svs_basename):
    if svs_basename in cluster_data:
        node = cluster_data[svs_basename]
        if not isinstance(node, dict):
            return {}
        
        flat = {}
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            if "count" in v:
                flat[k] = v
            else:
                for kk, vv in v.items():
                    if isinstance(vv, dict) and "count" in vv:
                        flat[kk] = vv
        return flat if flat else node
    
    for k in cluster_data.keys():
        if str(k) in svs_basename:
            return get_clusters_for_svs(cluster_data, k)
    return {}

def format_qa_output(qa_data, prefix="qa"):
    """格式化QA输出，将QA数据转换为带ID的字典格式"""
    if qa_data is None:
        return {}
        
    results = {}
    if isinstance(qa_data, list):
        for item in qa_data:
            if isinstance(item, dict):
                uid = generate_qa_id()
                item['id'] = uid
                results[uid] = item
    elif isinstance(qa_data, dict):
        if "question" in qa_data:
            uid = generate_qa_id()
            qa_data['id'] = uid
            results[uid] = qa_data
        else:
            return qa_data
    return results

def calculate_tissue_ratio(img: Image.Image) -> float:
    gray_img = img.convert('L')
    np_img = np.array(gray_img)
    tissue_pixels = np.count_nonzero(np_img < BACKGROUND_THRESHOLD)
    total_pixels = np_img.size
    return tissue_pixels / total_pixels if total_pixels > 0 else 0.0

# ============================================================
# IMAGE EXTRACTION FUNCTIONS
# ============================================================

def extract_and_pad_region(slide, x0, y0, w, h):
    """从slide提取区域并填充为正方形"""
    region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
    if w != h:
        m = max(w, h)
        canvas = Image.new("RGB", (m, m), (255, 255, 255))
        paste_x = (m - w) // 2
        paste_y = (m - h) // 2
        canvas.paste(region, (paste_x, paste_y))
        return canvas
    return region

def extract_cluster_image(slide, cluster, output_path):
    """提取cluster图像"""
    cx = int(cluster.get('center_x', 0))
    cy = int(cluster.get('center_y', 0))
    radius = int(float(cluster['radius']))
    
    L = radius * 2
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    
    max_w = slide.dimensions[0] - x0
    max_h = slide.dimensions[1] - y0
    w = min(L, max_w)
    h = min(L, max_h)
    
    region_sq = extract_and_pad_region(slide, x0, y0, w, h)
    
    tissue_ratio = calculate_tissue_ratio(region_sq)
    if tissue_ratio < MIN_TISSUE_RATIO:
        return None, tissue_ratio
    
    cluster_img = region_sq.resize((CLUSTER_OUTPUT_SIZE, CLUSTER_OUTPUT_SIZE), Image.LANCZOS)
    cluster_img.save(output_path)
    return cluster_img, tissue_ratio

def extract_roi_image(slide, cx, cy, mag, output_path):
    """提取ROI图像"""
    size = LAYER_PIXELS.get(mag, ROI_OUTPUT_SIZE)
    rx0 = max(0, cx - size // 2)
    ry0 = max(0, cy - size // 2)
    
    rx_w = min(size, slide.dimensions[0] - rx0)
    rx_h = min(size, slide.dimensions[1] - ry0)
    
    region_roi_sq = extract_and_pad_region(slide, rx0, ry0, rx_w, rx_h)
    roi_img = region_roi_sq.resize((ROI_OUTPUT_SIZE, ROI_OUTPUT_SIZE), Image.LANCZOS)
    roi_img.save(output_path)
    return roi_img

# ============================================================
# ASYNC WORKERS
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

async def task_infer_roi(img_path, prompt, client, sem):
    img_url = image_to_data_url(img_path)
    messages = [
        {"role":"system","content":"You are a professional hepatopathologist analyzing liver biopsy images."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_qa_generate(img_path, diagnosis, visual_context, q_type, difficulty, client, sem):
    prompt = get_qa_prompt(diagnosis, visual_context, q_type, difficulty)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role": "system", "content": "You are a pathology education assistant. Focus on visual evidence."},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=True)

async def task_caption_qa(img_path, diagnosis, visual_context, mag, client, sem):
    prompt = get_captionqa_prompt(diagnosis, visual_context, mag)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role":"system","content":"You are an expert pathology education assistant."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=True)

async def task_infer_cluster_image(cluster_id, cluster_img_path, roi_texts, diagnosis, client, sem):
    img_url = image_to_data_url(cluster_img_path)
    prompt = get_cluster_infer_prompt(cluster_id, roi_texts, diagnosis)
    messages = [
        {"role":"system","content":"You summarize cluster-level findings by integrating ROI inferences with the cluster image."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_infer_wsi_summary(thumb_path, cluster_summaries, diagnosis, client, sem):
    img_url = image_to_data_url(thumb_path)
    prompt = get_wsi_summary_prompt(cluster_summaries, diagnosis)
    messages = [
        {"role":"system","content":"You integrate cluster-level micro summaries with macro image."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

# ============================================================
# PROCESSING STAGES
# ============================================================

async def extract_images_and_run_roi_inference(slide, selected_clusters, roi_dir, cluster_img_dir, diagnosis_text, client, sem):
    """STEP 1: 提取ROI和Cluster图像，并运行ROI推理"""
    roi_data_map = []
    infer_tasks = []
    cluster_image_map = {}
    
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        cx = int(cluster.get('center_x', 0))
        cy = int(cluster.get('center_y', 0))
        
        cluster_img_path = os.path.join(cluster_img_dir, f"{cname}.png")
        cluster_img, tissue_ratio = extract_cluster_image(slide, cluster, cluster_img_path)
        
        if cluster_img is None:
            print(f"⚠ Skipping cluster {cname}: Tissue ratio {tissue_ratio:.1%} is below threshold {MIN_TISSUE_RATIO:.1%} (Mostly whitespace)")
            continue
        
        cluster_image_map[cname] = cluster_img_path
        
        for mag in MAGNIFICATIONS:
            roi_path = os.path.join(roi_dir, f"{cname}_{mag}x.png")
            extract_roi_image(slide, cx, cy, mag, roi_path)
            
            infer_prompt = get_infer_prompt(mag, diagnosis_text)
            t_infer = asyncio.create_task(task_infer_roi(roi_path, infer_prompt, client, sem))
            infer_tasks.append(t_infer)
            roi_data_map.append({
                "cluster": cname,
                "mag": mag,
                "path": roi_path,
                "diagnosis": diagnosis_text
            })
    
    # 使用异常处理，确保失败时取消所有任务
    try:
        inference_results = await asyncio.gather(*infer_tasks)
    except Exception as e:
        # 取消所有未完成的任务
        for task in infer_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"ROI inference failed: {e}") from e
    
    return roi_data_map, inference_results, cluster_image_map

async def _run_roi_qa_group(idx, m_path, m_diag, m_ctx, m_mag, client, sem):
    """并行生成单个ROI的所有QA（单选、多选、caption），所有API调用受semaphore限制"""
    # 创建所有QA生成任务（并行执行）
    tasks = []
    difficulty_tasks = []
    
    # 为每个难度级别创建单选和多选任务
    for d in DIFFICULTIES:
        difficulty_tasks.append(('single', d, task_qa_generate(m_path, m_diag, m_ctx, "single-choice", d, client, sem)))
        difficulty_tasks.append(('multi', d, task_qa_generate(m_path, m_diag, m_ctx, "multiple-choice", d, client, sem)))
    
    # Caption QA任务
    caption_task = task_caption_qa(m_path, m_diag, m_ctx, m_mag, client, sem)
    
    # 收集所有任务
    all_tasks = [t[2] for t in difficulty_tasks] + [caption_task]
    
    try:
        # 并行执行所有QA生成（每个API调用都会通过semaphore控制并发）
        results = await asyncio.gather(*all_tasks)
    except Exception as e:
        # 如果有任务失败，取消所有未完成的任务
        for task in all_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"ROI QA generation failed for ROI {idx}: {e}") from e
    
    # 解析结果
    qa_singles = {}
    qa_multis = {}
    
    result_idx = 0
    for qa_type, difficulty, _ in difficulty_tasks:
        if qa_type == 'single':
            qa_singles[difficulty] = results[result_idx]
        else:
            qa_multis[difficulty] = results[result_idx]
        result_idx += 1
    
    qa_caption = results[-1]
    
    return idx, qa_singles, qa_multis, qa_caption

async def generate_roi_qa_and_prepare_cluster_contexts(roi_data_map, inference_results, client, sem):
    """STEP 2: 生成ROI QA并准备cluster上下文"""
    qa_tasks = []
    cluster_texts_map = {}
    
    for i, meta in enumerate(roi_data_map):
        inf_res = inference_results[i]
        cname = meta['cluster']
        if cname not in cluster_texts_map:
            cluster_texts_map[cname] = []
        cluster_texts_map[cname].append(f"Mag {meta['mag']}x: {inf_res}")
        
        # 创建QA生成任务（所有ROI的QA会并行生成，但每个API调用都受semaphore限制）
        qa_tasks.append(_run_roi_qa_group(i, meta['path'], meta['diagnosis'], inf_res, meta['mag'], client, sem))
    
    return qa_tasks, cluster_texts_map

async def generate_cluster_summaries(selected_clusters, cluster_image_map, cluster_texts_map, diagnosis_text, client, sem):
    """STEP 2b: 生成cluster总结"""
    cluster_img_tasks = []
    cluster_names_ordered = []
    
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        if cname not in cluster_image_map:
            continue
        
        cluster_names_ordered.append(cname)
        cluster_img_path = cluster_image_map.get(cname)
        roi_texts = cluster_texts_map.get(cname, [])
        
        if cluster_img_path:
            cluster_img_tasks.append(
                asyncio.create_task(task_infer_cluster_image(cname, cluster_img_path, roi_texts, diagnosis_text, client, sem))
            )
        else:
            print(f"⚠ Skipping cluster image inference for {cname}: Missing image.")
            cluster_names_ordered.pop()
    
    try:
        cluster_img_sum_results = await asyncio.gather(*cluster_img_tasks)
    except Exception as e:
        # 取消所有未完成的任务
        for task in cluster_img_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"Cluster summary generation failed: {e}") from e
    
    cluster_summary_map = {}
    summary_list_for_wsi = []
    for cname, summary in zip(cluster_names_ordered, cluster_img_sum_results):
        cluster_summary_map[cname] = summary
        summary_list_for_wsi.append({"cluster_id": cname, "summary": summary})
    
    return cluster_summary_map, summary_list_for_wsi, cluster_names_ordered

async def _run_cluster_qa_group(cname, img_path, diag, ctx, client, sem):
    """并行生成单个Cluster的所有QA（单选、多选、caption），所有API调用受semaphore限制"""
    # 创建所有QA生成任务（并行执行）
    tasks = []
    difficulty_tasks = []
    
    # 为每个难度级别创建单选和多选任务
    for d in DIFFICULTIES:
        difficulty_tasks.append(('single', d, task_qa_generate(img_path, diag, ctx, "single-choice", d, client, sem)))
        difficulty_tasks.append(('multi', d, task_qa_generate(img_path, diag, ctx, "multiple-choice", d, client, sem)))
    
    # Caption QA任务
    caption_task = task_caption_qa(img_path, diag, ctx, 20, client, sem)
    
    # 收集所有任务
    all_tasks = [t[2] for t in difficulty_tasks] + [caption_task]
    
    try:
        # 并行执行所有QA生成（每个API调用都会通过semaphore控制并发）
        results = await asyncio.gather(*all_tasks)
    except Exception as e:
        # 如果有任务失败，取消所有未完成的任务
        for task in all_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"Cluster QA generation failed for {cname}: {e}") from e
    
    # 解析结果
    qa_singles = {}
    qa_multis = {}
    
    result_idx = 0
    for qa_type, difficulty, _ in difficulty_tasks:
        if qa_type == 'single':
            qa_singles[difficulty] = results[result_idx]
        else:
            qa_multis[difficulty] = results[result_idx]
        result_idx += 1
    
    qa_caption = results[-1]
    
    return cname, qa_singles, qa_multis, qa_caption

async def generate_cluster_qa(selected_clusters, cluster_image_map, cluster_summary_map, diagnosis_text, client, sem):
    """STEP 2c: 生成cluster级别QA"""
    
    cluster_qa_tasks = []
    
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        if cname not in cluster_image_map:
            continue
        
        summary = cluster_summary_map.get(cname, "")
        cluster_img_path = cluster_image_map.get(cname)
        
        if cluster_img_path and summary:
            cluster_qa_tasks.append(_run_cluster_qa_group(cname, cluster_img_path, diagnosis_text, summary, client, sem))
        else:
            print(f"⚠ Skipping QA for cluster {cname}: Missing image or summary.")
    
    try:
        cluster_qa_results_list = await asyncio.gather(*cluster_qa_tasks)
    except Exception as e:
        # 取消所有未完成的任务
        for task in cluster_qa_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"Cluster QA generation failed: {e}") from e
    
    cluster_qa_results_map = {}
    for cname, res_s_map, res_m_map, res_c in cluster_qa_results_list:
        processed_singles = {}
        for d in DIFFICULTIES:
            processed_singles.update(format_qa_output(res_s_map[d]))
        
        processed_multis = {}
        for d in DIFFICULTIES:
            processed_multis.update(format_qa_output(res_m_map[d]))
        
        cluster_qa_results_map[cname] = {
            "qa_single": processed_singles,
            "qa_multi": processed_multis,
            "qa_caption": format_qa_output(res_c)
        }
    
    return cluster_qa_results_map

async def generate_wsi_summary_and_qa(thumb_path, summary_list_for_wsi, diagnosis_text, client, sem):
    """STEP 3: 生成WSI总结和QA（并行生成所有QA）"""
    # 先生成WSI总结
    wsi_summary_text = await task_infer_wsi_summary(thumb_path, summary_list_for_wsi, diagnosis_text, client, sem)
    
    # 并行生成所有WSI QA
    qa_tasks = []
    difficulty_tasks = []
    
    for d in DIFFICULTIES:
        difficulty_tasks.append(('single', d, task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "single-choice", d, client, sem)))
        difficulty_tasks.append(('multi', d, task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "multiple-choice", d, client, sem)))
    
    caption_task = task_caption_qa(thumb_path, diagnosis_text, wsi_summary_text, 20, client, sem)
    
    all_tasks = [t[2] for t in difficulty_tasks] + [caption_task]
    
    try:
        results = await asyncio.gather(*all_tasks)
    except Exception as e:
        # 取消所有未完成的任务
        for task in all_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"WSI QA generation failed: {e}") from e
    
    # 解析结果
    wsi_qa_s_map = {}
    wsi_qa_m_map = {}
    
    result_idx = 0
    for qa_type, difficulty, _ in difficulty_tasks:
        if qa_type == 'single':
            wsi_qa_s_map[difficulty] = results[result_idx]
        else:
            wsi_qa_m_map[difficulty] = results[result_idx]
        result_idx += 1
    
    wsi_cap = results[-1]
    
    processed_wsi_s = {}
    processed_wsi_m = {}
    for d in DIFFICULTIES:
        processed_wsi_s.update(format_qa_output(wsi_qa_s_map[d]))
        processed_wsi_m.update(format_qa_output(wsi_qa_m_map[d]))
    
    return wsi_summary_text, processed_wsi_s, processed_wsi_m, wsi_cap

def assemble_final_results(roi_data_map, inference_results, roi_results_staging, cluster_image_map, 
                          cluster_summary_map, cluster_qa_results_map, thumb_path, wsi_path, 
                          wsi_summary_text, processed_wsi_s, processed_wsi_m, wsi_cap, timestamp, diagnosis_text):
    """STEP 4: 组装最终结果"""
    
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
        
        final_clusters_map[cname]["rois"].append({
            "magnification": meta['mag'],
            "image_path": meta['path'],
            "inference_result": inference_results[i],
            "qa_single_choice": roi_results_staging[i]["qa_single"],
            "qa_multiple_choice": roi_results_staging[i]["qa_multi"],
            "caption_qa": roi_results_staging[i]["qa_caption"]
        })
    
    final_output = {
        "session_info": {
            "timestamp": timestamp,
            "wsi_path": wsi_path,
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
    
    return final_output

# ============================================================
# MAIN PIPELINE
# ============================================================

async def main(wsi_path, case_info_json, cluster_geometry, output_dir):
    """主处理流程，如果遇到 API 错误会抛出异常"""
    # 初始化
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
    session_dir = os.path.join(output_dir, f"session_{svs_basename}_{timestamp}")
    roi_dir = os.path.join(session_dir, "rois")
    cluster_img_dir = os.path.join(session_dir, "clusters")
    ensure_dir(session_dir)
    ensure_dir(roi_dir)
    ensure_dir(cluster_img_dir)
    
    if not os.path.exists(wsi_path):
        print(f"❌ Error: WSI file not found at {wsi_path}")
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")

    slide = OpenSlide(wsi_path)
    diagnosis_text = get_case_diagnosis(wsi_path, case_info_json)
    cluster_raw = load_json(cluster_geometry)
    

    thumb = slide.get_thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    thumb_path = os.path.join(session_dir, "thumbnail.png")
    thumb.save(thumb_path)

    clusters = get_clusters_for_svs(cluster_raw, svs_basename)
    selected_clusters = []
    for k, v in clusters.items():
        if v.get("count", 0) > CLUSTER_COUNT_THRESHOLD:
            v['cluster_name'] = k
            selected_clusters.append(v)

    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # STEP 1: 提取图像并运行ROI推理
    roi_data_map, inference_results, cluster_image_map = await extract_images_and_run_roi_inference(
        slide, selected_clusters, roi_dir, cluster_img_dir, diagnosis_text, client, sem
    )
    print(f"Step1: Selected {len(selected_clusters)} clusters")

    # STEP 2: 生成ROI QA并准备cluster上下文
    qa_tasks, cluster_texts_map = await generate_roi_qa_and_prepare_cluster_contexts(
        roi_data_map, inference_results, client, sem
    )
    print(f"Step2: Generated {len(qa_tasks)} ROI QA tasks")
    # STEP 2b: 生成cluster总结
    cluster_summary_map, summary_list_for_wsi, cluster_names_ordered = await generate_cluster_summaries(
        selected_clusters, cluster_image_map, cluster_texts_map, diagnosis_text, client, sem
    )
    print(f"Step2b: Generated {len(cluster_summary_map)} cluster summaries")
    try:
        roi_qa_results_list = await asyncio.gather(*qa_tasks)
    except Exception as e:
        # 取消所有未完成的任务
        for task in qa_tasks:
            if not task.done():
                task.cancel()
        raise RuntimeError(f"ROI QA processing failed: {e}") from e

    roi_results_staging = [None] * len(roi_data_map)
    for idx, res_s_map, res_m_map, res_c in roi_qa_results_list:
        processed_singles = {}
        for d in DIFFICULTIES:
            processed_singles.update(format_qa_output(res_s_map[d]))
        
        processed_multis = {}
        for d in DIFFICULTIES:
            processed_multis.update(format_qa_output(res_m_map[d]))
        
        roi_results_staging[idx] = {
            "qa_single": processed_singles,
            "qa_multi": processed_multis,
            "qa_caption": format_qa_output(res_c)
        }

    # STEP 2c: 生成cluster级别QA
    cluster_qa_results_map = await generate_cluster_qa(
        selected_clusters, cluster_image_map, cluster_summary_map, diagnosis_text, client, sem
    )
    print(f"Step2c: Generated {len(cluster_qa_results_map)} cluster QA results")
    # STEP 3: 生成WSI总结和QA
    wsi_summary_text, processed_wsi_s, processed_wsi_m, wsi_cap = await generate_wsi_summary_and_qa(
        thumb_path, summary_list_for_wsi, diagnosis_text, client, sem
    )
    print(f"Step3: Generated WSI summary and QA")
    # STEP 4: 组装最终结果
    final_output = assemble_final_results(
        roi_data_map, inference_results, roi_results_staging, cluster_image_map,
        cluster_summary_map, cluster_qa_results_map, thumb_path, wsi_path,
        wsi_summary_text, processed_wsi_s, processed_wsi_m, wsi_cap, timestamp, diagnosis_text
    )
    out_json_path = os.path.join(session_dir, "final_integrated_results.json")
    save_json(final_output, out_json_path)
    print(f"All done! Results saved to {out_json_path}")


def cleanup_incomplete_session(output_dir, wsi_path):
    """清理未完成的session目录"""
    try:
        svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
        pattern = os.path.join(output_dir, f"session_{svs_basename}_*")
        incomplete_dirs = glob.glob(pattern)
        
        for incomplete_dir in incomplete_dirs:
            # 检查是否有最终结果文件，如果没有则认为是未完成的
            final_json = os.path.join(incomplete_dir, "final_integrated_results.json")
            if not os.path.exists(final_json):
                try:
                    shutil.rmtree(incomplete_dir)
                    print(f"🗑️  Cleaned up incomplete session: {incomplete_dir}")
                except Exception as cleanup_err:
                    print(f"⚠ Failed to cleanup {incomplete_dir}: {cleanup_err}")
    except Exception as e:
        print(f"⚠ Cleanup process encountered error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理WSI图像并生成诊断报告和QA")
    parser.add_argument("--wsi-path", type=str, required=True,
                        help="WSI文件路径")
    parser.add_argument("--case-info-json", type=str, required=True,
                        help="病例信息JSON文件路径")
    parser.add_argument("--cluster-geometry", type=str, required=True,
                        help="Cluster几何信息JSON文件路径")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(main(
            wsi_path=args.wsi_path,
            case_info_json=args.case_info_json,
            cluster_geometry=args.cluster_geometry,
            output_dir=args.output_dir
        ))
        print("✅ Processing completed successfully")
        sys.exit(0)
    except RuntimeError as e:
        print(f"❌ Fatal error: {e}")
        print("⚠ This WSI will NOT be marked as completed and can be retried later")
        # 清理未完成的session目录
        cleanup_incomplete_session(args.output_dir, args.wsi_path)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("⚠ This WSI will NOT be marked as completed and can be retried later")
        # 清理未完成的session目录
        cleanup_incomplete_session(args.output_dir, args.wsi_path)
        sys.exit(1)