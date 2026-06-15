#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import json
import base64
import asyncio
import uuid
import re
import argparse
import shutil
import glob
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

from PIL import Image, ImageStat
from openslide import OpenSlide
from openai import OpenAI

# 导入提示词模块 (请确保 wsi_bench_prompt.py 在同级目录下)
from prompt.wsi_bench_prompt import (
    get_vqa_prompt,
    get_infer_prompt,
    get_cluster_infer_prompt,
    get_wsi_infer_prompt,
    VQA_CATEGORIES
)

# ============================================================
# I. 配置与全局常量
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

# 运行控制
MAX_CONCURRENT_REQUESTS = 5
CLUSTER_COUNT_THRESHOLD = 100  # 忽略过小的 Cluster

# API 指数退避重试配置
MAX_RETRIES = 5  
INITIAL_RETRY_DELAY = 1  
MAX_RETRY_DELAY = 60  

LEVEL_TASK_MATRIX = {
    "WSI": ["1.1", "1.2", "2.1", "2.4"],
    "Cluster": ["1.2", "1.3", "2.1", "2.2"],
    "ROI": ["1.3", "1.4", "2.1", "2.2", "2.3"]
}

# ============================================================
# II. 辅助工具函数
# ============================================================

def load_json(p: str) -> Any:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj: Any, p: str):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def image_to_data_url(image_path: str) -> str:
    """将图像转换为 Base64 Data URL"""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def is_tissue(image: Image.Image, threshold: int = 220) -> bool:
    """检测是否为组织区域，避免在空白区域浪费 API"""
    stat = ImageStat.Stat(image.convert("L"))
    return stat.mean[0] < threshold

def try_parse_json(text: str) -> Any:
    """鲁棒的 JSON 解析，处理 LLM 可能多出的 Markdown 标记"""
    try:
        clean_text = re.sub(r"```json\s*|\s*```", "", text).strip()
        return json.loads(clean_text)
    except:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        return {"raw_text": text, "error": "JSON parse failed"}

def get_case_metadata(wsi_path: str, case_json_path: str) -> Dict:
    cases = load_json(case_json_path)
    if isinstance(cases, dict): cases = [cases]
    target_name = os.path.basename(wsi_path)
    metadata = {"diagnosis": "N/A", "ihc": "N/A", "ptnm": "N/A"}
    for c in cases:
        if os.path.splitext(c.get("img", ""))[0] in target_name:
            metadata.update(c)
            break
    return metadata

# ============================================================
# III. 核心推理任务 (Task-specific Functions)
# ============================================================

async def gpt_call_safe(messages: List[Dict], client: OpenAI, sem: asyncio.Semaphore, output_json: bool = False):
    """带指数退避重试机制的异步调用"""
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=API_MODEL,
                    messages=messages,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                return try_parse_json(content) if output_json else content
            except Exception as e:
                delay = min(MAX_RETRY_DELAY, INITIAL_RETRY_DELAY * (2 ** attempt))
                print(f"  [⚠️ Retry {attempt+1}] Delay {delay}s | Error: {e}")
                await asyncio.sleep(delay)
        return None

async def task_infer_wsi_summary(thumb_path: str, diagnosis: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    prompt = get_wsi_infer_prompt(diagnosis) 
    img_url = image_to_data_url(thumb_path)
    messages = [
        {"role":"system","content":"You summarize WSI macro findings."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_infer_cluster_image(cluster_id: str, cluster_img_path: str, wsi_context: str, diagnosis: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    prompt = get_cluster_infer_prompt(cluster_id, wsi_context, diagnosis)
    img_url = image_to_data_url(cluster_img_path)
    messages = [
        {"role":"system","content":"You summarize cluster-level findings by integrating WSI context with the cluster image."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def task_infer_roi(img_path: str, mag: int, diagnosis: str, cluster_context: str, wsi_context: str, client: OpenAI, sem: asyncio.Semaphore) -> str:
    prompt = get_infer_prompt(mag, diagnosis, cluster_context, wsi_context)
    img_url = image_to_data_url(img_path)
    messages = [
        {"role":"system","content":"You are a professional hepatopathologist analyzing liver biopsy images."},
        {"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":img_url}}]}
    ]
    return await gpt_call_safe(messages, client, sem, output_json=False)

async def generate_vqa_set(level_type: str, img_path: str, metadata: Dict, summary: str, mag: str, client: OpenAI, sem: asyncio.Semaphore):
    codes = LEVEL_TASK_MATRIX.get(level_type, [])
    sub_types = ["single", "multi", "caption"]
    ratio = {"single": 2, "multi": 1, "caption": 2}
    img_url = image_to_data_url(img_path)

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

    final_qas = []
    for code, stype, msgs in tasks:
        output = await gpt_call_safe(msgs, client, sem, output_json=True)
        if output:
            items = output if isinstance(output, list) else [output]
            for item in items:
                if isinstance(item, dict) and (item.get("question") or item.get("caption")):
                    item.update({"subcategory": code, "format": stype, "uuid": str(uuid.uuid4())})
                    final_qas.append(item)
    return final_qas

# ============================================================
# IV. 主流程管道
# ============================================================

async def run_pipeline(wsi_path: str, case_json: str, geometry_json: str, output_root: str):
    svs_id = os.path.splitext(os.path.basename(wsi_path))[0]
    save_dir = os.path.join(output_root, svs_id)
    os.makedirs(save_dir, exist_ok=True)
    
    slide = OpenSlide(wsi_path)
    metadata = get_case_metadata(wsi_path, case_json)
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    final_output = {"wsi_id": svs_id, "results": []}

    # --- Step 1: WSI 层 ---
    print(f"[*] Level WSI: {svs_id}")
    wsi_thumb_path = os.path.join(save_dir, "wsi_thumb.png")
    slide.get_thumbnail((2048, 2048)).save(wsi_thumb_path)
    
    wsi_summary = await task_infer_wsi_summary(wsi_thumb_path, metadata["diagnosis"], client, sem)
    wsi_vqa = await generate_vqa_set("WSI", wsi_thumb_path, metadata, wsi_summary, "Macro", client, sem)
    
    final_output["results"].append({
        "level": "WSI", "summary": wsi_summary, "vqa": wsi_vqa, "image_path": wsi_thumb_path
    })

    # --- Step 2: Cluster 层 ---
    geometry_data = load_json(geometry_json)
    top_key = list(geometry_data.keys())[0]
    clusters_dict = geometry_data[top_key]
    
    for c_key, c_info in clusters_dict.items():
        # Cluster 阈值过滤
        if c_info.get("count", 0) < CLUSTER_COUNT_THRESHOLD:
            continue

        print(f"  [+] Processing {c_key} (Size: {c_info.get('count')})...")
        cx, cy, r = c_info["center_x"], c_info["center_y"], c_info["radius"]
        side = int(r * 1.5)
        
        c_img = slide.read_region((max(0, int(cx - side//2)), max(0, int(cy - side//2))), 0, (side, side)).convert("RGB")
        if not is_tissue(c_img): continue
        
        c_img.thumbnail((2048, 2048))
        c_path = os.path.join(save_dir, f"{c_key}.png")
        c_img.save(c_path)
        
        c_summary = await task_infer_cluster_image(c_key, c_path, wsi_summary, metadata["diagnosis"], client, sem)
        c_vqa = await generate_vqa_set("Cluster", c_path, metadata, c_summary, "Medium", client, sem)
        
        cluster_entry = {
            "level": "Cluster", "id": c_key, "summary": c_summary, 
            "vqa": c_vqa, "image_path": c_path, "rois": []
        }

        # --- Step 3: ROI 层 ---
        for mag in [10, 20]:
            sz = 2048 if mag == 10 else 1024
            roi_img = slide.read_region((int(cx - sz // 2), int(cy - sz // 2)), 0, (sz, sz)).convert("RGB")
            if not is_tissue(roi_img): continue
            
            roi_img.thumbnail((1024, 1024))
            r_path = os.path.join(save_dir, f"{c_key}_roi_{mag}x.png")
            roi_img.save(r_path)

            r_summary = await task_infer_roi(r_path, mag, metadata["diagnosis"], c_summary, wsi_summary, client, sem)
            r_vqa = await generate_vqa_set("ROI", r_path, metadata, r_summary, f"{mag}x", client, sem)

            cluster_entry["rois"].append({
                "mag": mag, "summary": r_summary, "vqa": r_vqa, "image_path": r_path
            })

        final_output["results"].append(cluster_entry)

    save_json(final_output, os.path.join(save_dir, "final_vqa_dataset.json"))
    print(f"✅ Finished WSI: {svs_id}")

# ============================================================
# V. 程序入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi-path", type=str, required=True)
    parser.add_argument("--case-info-json", type=str, required=True)
    parser.add_argument("--cluster-geometry", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    try:
        asyncio.run(run_pipeline(args.wsi_path, args.case_info_json, args.cluster_geometry, args.output_dir))
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        sys.exit(1)