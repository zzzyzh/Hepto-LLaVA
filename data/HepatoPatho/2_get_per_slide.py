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
MAX_CONCURRENT_REQUESTS = 9

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
    if qa_data is None or isinstance(qa_data, dict) and qa_data.get('error'):
        return qa_data if qa_data else {"error": "Empty API response"}
        
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

async def gpt_call_safe(messages, client, semaphore, output_json=False):
    async with semaphore:
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
            print(f"⚠ API Error: {e}")
            return {"error": str(e)} if output_json else f"Error: {str(e)}"

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
    
    inference_results = await asyncio.gather(*infer_tasks)
    
    return roi_data_map, inference_results, cluster_image_map

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
        
        async def run_roi_qa_group(idx, m_path, m_diag, m_ctx, m_mag):
            qa_singles = {}
            qa_multis = {}
            for d in DIFFICULTIES:
                t_s = await task_qa_generate(m_path, m_diag, m_ctx, "single-choice", d, client, sem)
                t_m = await task_qa_generate(m_path, m_diag, m_ctx, "multiple-choice", d, client, sem)
                qa_singles[d] = t_s
                qa_multis[d] = t_m
            t_c = await task_caption_qa(m_path, m_diag, m_ctx, m_mag, client, sem)
            return idx, qa_singles, qa_multis, t_c
        
        qa_tasks.append(run_roi_qa_group(i, meta['path'], meta['diagnosis'], inf_res, meta['mag']))
    
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
    
    cluster_img_sum_results = await asyncio.gather(*cluster_img_tasks)
    
    cluster_summary_map = {}
    summary_list_for_wsi = []
    for cname, summary in zip(cluster_names_ordered, cluster_img_sum_results):
        cluster_summary_map[cname] = summary
        summary_list_for_wsi.append({"cluster_id": cname, "summary": summary})
    
    return cluster_summary_map, summary_list_for_wsi, cluster_names_ordered

async def generate_cluster_qa(selected_clusters, cluster_image_map, cluster_summary_map, diagnosis_text, client, sem):
    """STEP 2c: 生成cluster级别QA"""
    
    cluster_qa_tasks = []
    
    async def run_cluster_qa_group(cname, img_path, diag, ctx):
        qa_singles = {}
        qa_multis = {}
        for d in DIFFICULTIES:
            t_s = await task_qa_generate(img_path, diag, ctx, "single-choice", d, client, sem)
            t_m = await task_qa_generate(img_path, diag, ctx, "multiple-choice", d, client, sem)
            qa_singles[d] = t_s
            qa_multis[d] = t_m
        t_c = await task_caption_qa(img_path, diag, ctx, 20, client, sem)
        return cname, qa_singles, qa_multis, t_c
    
    for cluster in selected_clusters:
        cname = cluster.get('cluster_name')
        if cname not in cluster_image_map:
            continue
        
        summary = cluster_summary_map.get(cname, "")
        cluster_img_path = cluster_image_map.get(cname)
        
        if cluster_img_path and summary:
            cluster_qa_tasks.append(run_cluster_qa_group(cname, cluster_img_path, diagnosis_text, summary))
        else:
            print(f"⚠ Skipping QA for cluster {cname}: Missing image or summary.")
    
    cluster_qa_results_list = await asyncio.gather(*cluster_qa_tasks)
    
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
    """STEP 3: 生成WSI总结和QA"""
    wsi_summary_text = await task_infer_wsi_summary(thumb_path, summary_list_for_wsi, diagnosis_text, client, sem)
    
    wsi_qa_s_map = {}
    wsi_qa_m_map = {}
    
    for d in DIFFICULTIES:
        wsi_qa_s_map[d] = await task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "single-choice", d, client, sem)
        wsi_qa_m_map[d] = await task_qa_generate(thumb_path, diagnosis_text, wsi_summary_text, "multiple-choice", d, client, sem)
    
    wsi_cap = await task_caption_qa(thumb_path, diagnosis_text, wsi_summary_text, 20, client, sem)
    
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
        return

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
    roi_qa_results_list = await asyncio.gather(*qa_tasks)

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
    
    asyncio.run(main(
        wsi_path=args.wsi_path,
        case_info_json=args.case_info_json,
        cluster_geometry=args.cluster_geometry,
        output_dir=args.output_dir
    ))