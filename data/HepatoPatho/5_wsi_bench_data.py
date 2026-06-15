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
import random
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

from PIL import Image, ImageStat
from openslide import OpenSlide
from openai import OpenAI

# 导入您提供的提示词模块
from prompt.wsi_bench_prompt import (
    get_vqa_prompt,
    get_infer_prompt,
    get_cluster_infer_prompt,
    get_wsi_infer_prompt,
    VQA_CATEGORIES
)

# ============================================================
# I. 配置与初始化
# ============================================================

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

MAX_CONCURRENT_REQUESTS = 5
MAX_RETRIES = 3

LEVEL_TASK_MATRIX = {
    "WSI": ["1.1", "1.2", "2.1", "2.4"],
    "Cluster": ["1.2", "1.3", "2.1", "2.2"],
    "ROI": ["1.3", "1.4", "2.1", "2.2", "2.3"]
}

# ============================================================
# II. 辅助函数 (全量补全)
# ============================================================

def load_json(p: str) -> Any:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_json(obj: Any, p: str):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def is_tissue(image: Image.Image, threshold: int = 220) -> bool:
    """检测是否为组织区域，阈值设为 220 避免漏掉浅色组织"""
    stat = ImageStat.Stat(image.convert("L"))
    return stat.mean[0] < threshold

def get_case_metadata(wsi_path: str, case_json_path: str) -> Dict:
    cases = load_json(case_json_path)
    if isinstance(cases, dict): cases = [cases]
    target_name = os.path.basename(wsi_path)
    metadata = {"diagnosis": "N/A",  "ihc": "N/A", "ptnm": "N/A"}
    
    for c in cases:
        if os.path.splitext(c.get("img", ""))[0] in target_name:
            metadata.update(c)
            diag_text = str(c.get("diagnosis", ""))
            if "IHC:" in diag_text:
                metadata["ihc"] = diag_text.split("IHC:")[1].split(".")[0].strip()
            ptnm_match = re.search(r"pT[0-4]N[0-3x][a-z0-9]*", diag_text)
            if ptnm_match:
                metadata["ptnm"] = ptnm_match.group(0)
            break
    return metadata

def try_parse_json(text: str) -> Any:
    try:
        clean_text = re.sub(r"```json\s*|\s*```", "", text).strip()
        return json.loads(clean_text)
    except:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        return {"raw_text": text, "error": "JSON parse failed"}

def cleanup_incomplete_session(output_dir, wsi_path):
    try:
        svs_basename = os.path.splitext(os.path.basename(wsi_path))[0]
        pattern = os.path.join(output_dir, f"{svs_basename}")
        incomplete_dirs = glob.glob(pattern)
        for incomplete_dir in incomplete_dirs:
            final_json = os.path.join(incomplete_dir, "final_vqa_dataset.json")
            if not os.path.exists(final_json):
                shutil.rmtree(incomplete_dir)
                print(f"🗑️ Cleaned up incomplete session: {incomplete_dir}")
    except Exception as e:
        print(f"⚠ Cleanup error: {e}")

# ============================================================
# III. 异步 API 调用
# ============================================================

async def gpt_call(messages: List[Dict], client: OpenAI, semaphore: asyncio.Semaphore, is_json: bool = False):
    async with semaphore:
        for i in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=API_MODEL,
                    messages=messages,
                    temperature=0.7
                )
                content = response.choices[0].message.content
                return try_parse_json(content) if is_json else content
            except Exception as e:
                print(f"  [API Error] {e}")
                await asyncio.sleep(5)
        return None

async def generate_vqa_set(level_type: str, img_path: str, metadata: Dict, summary: str, mag: str, client: OpenAI, semaphore: asyncio.Semaphore):
    codes = LEVEL_TASK_MATRIX.get(level_type, [])  # 获取该层次对应的任务代码
    sub_types = ["single", "multi", "caption"]  # 问题类型：单选、多个选择、标题

    # 设置比例，控制每种问题类型生成的数量
    ratio = {"single": 2, "multi": 1, "caption": 2}
    total = sum(ratio.values())  # 总问题数
    base64_img = encode_image_base64(img_path)  # 将图像转换为 Base64 编码

    tasks = []
    for code in codes:  # 遍历该层次的任务代码
        for stype in sub_types:  # 遍历问题类型
            # 每个任务代码都根据比例生成问题
            for _ in range(ratio[stype]):
                prompt = get_vqa_prompt(code, stype, metadata, summary, mag)  # 根据任务代码和问题类型生成提示词
                msgs = [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]}]
                tasks.append((code, stype, msgs))  # 将任务添加到任务列表

    results = []  # 存储生成的问答
    for code, stype, msgs in tasks:  # 遍历任务并生成问题
        result = await gpt_call(msgs, client, semaphore, is_json=True)
        if result:
            results.append(result)

    final_qas = []  # 存储最终的问答对
    for idx, output in enumerate(results):
        code, stype, _ = tasks[idx]
        if output:
            items = output if isinstance(output, list) else [output]
            for item in items:
                if isinstance(item, dict) and (item.get("question") or item.get("caption")):
                    item.update({"subcategory": code, "format": stype})
                    # 为每个问题-答案对增加唯一的UUID
                    item["uuid"] = str(uuid.uuid4())
                    final_qas.append(item)

    return final_qas


# ============================================================
# IV. 主管道：处理嵌套字典结构的 Cluster
# ============================================================

async def run_pipeline(wsi_path: str, case_json: str, geometry_json: str, output_root: str):
    svs_id = os.path.splitext(os.path.basename(wsi_path))[0]
    save_dir = os.path.join(output_root, svs_id)
    os.makedirs(save_dir, exist_ok=True)
    
    slide = OpenSlide(wsi_path)
    metadata = get_case_metadata(wsi_path, case_json)
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    
    final_output = {"wsi_id": svs_id, "results": []}

    # --- Step 1: WSI 层 ---
    print(f"[*] Level WSI: {svs_id}")
    wsi_img = os.path.join(save_dir, "wsi_thumb.png")
    slide.get_thumbnail((2048, 2048)).save(wsi_img)
    wsi_infer = await gpt_call([{"role":"user","content":get_wsi_infer_prompt(metadata["diagnosis"])}], client, semaphore)
    wsi_vqa = await generate_vqa_set("WSI", wsi_img, metadata, wsi_infer, "Macro", client, semaphore)
    
    # Save WSI image path in JSON
    final_output["results"].append({
        "level": "WSI", 
        "summary": wsi_infer, 
        "vqa": wsi_vqa, 
        "image_path": wsi_img  # Save the image path
    })

    # --- Step 2: Cluster 层 ---
    geometry_data = load_json(geometry_json)
    top_key = list(geometry_data.keys())[0]
    clusters_dict = geometry_data[top_key]
    
    print(f"[*] Found {len(clusters_dict)} clusters in key '{top_key}'")

    for c_key, c_info in clusters_dict.items():
        print(f"  [+] Processing {c_key}...")
        cx, cy, r = c_info["center_x"], c_info["center_y"], c_info["radius"]
        
        # 计算 Cluster 矩形区域 (用于 read_region)
        side = int(r * 1.5)  # 取半径的 1.5 倍作为观察窗
        left = max(0, int(cx - side//2))
        top = max(0, int(cy - side//2))
        
        c_img = slide.read_region((left, top), 0, (side, side)).convert("RGB")
        c_img.thumbnail((2048, 2048))
        c_path = os.path.join(save_dir, f"{c_key}.png")
        c_img.save(c_path)
        
        c_infer = await gpt_call([{"role":"user","content":get_cluster_infer_prompt(c_key, wsi_infer, metadata["diagnosis"])}], client, semaphore)
        c_vqa = await generate_vqa_set("Cluster", c_path, metadata, c_infer, "Medium", client, semaphore)
        
        # Save Cluster image path in JSON
        cluster_entry = {
            "level": "Cluster", 
            "id": c_key, 
            "summary": c_infer, 
            "vqa": c_vqa, 
            "image_path": c_path,  # Save the image path for the cluster
            "rois": []
        }

        # --- Step 3: ROI 层 (使用 Cluster 中心点采样 10x 和 20x 放大倍数) ---
        print(f"    [-] Sampling ROIs for {c_key}...")
        roi_samples = []
        for mag in [10, 20]:  # 使用 10x 和 20x 放大倍数
            sz = 2048 if mag == 10 else 1024  # 设置图像尺寸
            roi_img = slide.read_region((int(cx - sz // 2), int(cy - sz // 2)), 0, (sz, sz)).convert("RGB")
            roi_img.thumbnail((1024, 1024))  # 调整尺寸为 1024x1024
            
            # 保存 ROI 图像
            r_path = os.path.join(save_dir, f"{c_key}_roi_{mag}x.png")
            roi_img.save(r_path)

            # 进行 ROI 推理
            r_infer = await gpt_call([{"role": "user", "content": get_infer_prompt(mag, metadata["diagnosis"], c_infer, wsi_infer)}], client, semaphore)
            
            # 生成 ROI 的 VQA
            r_vqa = await generate_vqa_set("ROI", r_path, metadata, r_infer, f"{mag}x", client, semaphore)

            # 将该 ROI 加入到 Cluster 的结果中
            roi_samples.append({"mag": mag, "summary": r_infer, "vqa": r_vqa, "image_path": r_path})  # Save the image path for each ROI

        # 添加 ROI 数据
        cluster_entry["rois"] = roi_samples

        # 将 cluster 数据添加到最终输出
        final_output["results"].append(cluster_entry)

    # 保存最终的结果（包括所有图像路径）
    save_json(final_output, os.path.join(save_dir, "final_vqa_dataset.json"))
    print(f"DONE. Output: {save_dir}")


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
        cleanup_incomplete_session(args.output_dir, args.wsi_path)
        sys.exit(1)