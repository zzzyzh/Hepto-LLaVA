import os
import json
import uuid
import base64
import mimetypes
import re
import asyncio
from typing import Dict, Any, List
from openai import OpenAI

from prompt.caption import (
    get_multilevel_caption_prompt
)

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

# API 重试配置
MAX_RETRIES = 5  # 最大重试次数
INITIAL_RETRY_DELAY = 1  # 初始重试延迟（秒）
MAX_RETRY_DELAY = 60  # 最大重试延迟（秒）
MAX_CONCURRENT_REQUESTS = 9  # 最大并发 API 请求数



def load_json(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def image_to_data_url(path: str) -> str:
    """将图像文件转换为 Base64 数据 URL，用于 API 调用。"""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        enc = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{enc}"

def generate_caption_id() -> str:
    """生成唯一的 caption ID。"""
    return f"caption_{uuid.uuid4().hex[:8]}"

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
    """
    根据 WSI 文件名（不含扩展名）在 case_json 中精确查找匹配的病例。
    如果未找到，抛出 ValueError。
    """
    cases = load_json(case_json_path)
    if isinstance(cases, dict):
        cases = [cases]
    
    # 提取 WSI ID（不含扩展名）
    wsi_basename = os.path.basename(wsi_path)
    wsi_id = os.path.splitext(wsi_basename)[0]  # e.g., "TCGA-DD-A1HB"

    # 精确匹配：case["img"] 去掉扩展名后必须等于 wsi_id
    for c in cases:
        img_field = c.get("img", "")
        if not isinstance(img_field, str):
            continue
        case_img_id = os.path.splitext(img_field)[0]
        if case_img_id == wsi_id:
            # 找到匹配项，返回完整元数据（缺失字段可设默认值）
            metadata = {
                "diagnosis": c.get("diagnosis", "N/A"),
            }
            return metadata

    # ❌ 未找到匹配项：明确报错并终止
    raise ValueError(
        f"WSI ID '{wsi_id}' (from '{wsi_path}') not found in case info JSON '{case_json_path}'. "
        f"Please ensure the 'img' field in case JSON matches the WSI filename (without extension)."
    )

# ------------------------------------------------------------
# 新增：Caption 解析器
# ------------------------------------------------------------
def parse_caption_abc(raw_output: str) -> Dict[str, str]:
    raw_lower = raw_output.lower()
    pattern = r'caption\s+([abc]):\s*(.*?)(?=caption\s+[abc]:|$)'
    matches = re.findall(pattern, raw_lower, re.DOTALL | re.IGNORECASE)
    result = {"caption_a": "", "caption_b": "", "caption_c": ""}
    
    for letter, _ in matches:
        key = f"caption_{letter.lower()}"
        # 在原始文本中精确定位
        start_marker_variants = [
            f"Caption {letter.upper()}:",
            f"caption {letter}:",
            f"CAPTION {letter.upper()}:",
            f"Caption {letter}:"
        ]
        actual_start = -1
        matched_len = 0
        for var in start_marker_variants:
            pos = raw_output.find(var)
            if pos != -1:
                actual_start = pos + len(var)
                matched_len = len(var)
                break
        
        if actual_start == -1:
            continue

        # 找下一个 caption 起始
        next_pos = len(raw_output)
        for next_letter in ['A', 'B', 'C']:
            if next_letter == letter.upper():
                continue
            for var in [f"Caption {next_letter}:", f"caption {next_letter.lower()}:", f"CAPTION {next_letter}:"]:
                p = raw_output.find(var, actual_start)
                if p != -1 and p < next_pos:
                    next_pos = p
        content = raw_output[actual_start:next_pos].strip()
        content = re.sub(r'^\s+', '', content)
        result[key] = content
    return result

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


# ------------------------------------------------------------
# 核心：为单图生成 caption
# ------------------------------------------------------------
async def generate_caption_for_image(
    image_id: str,
    gemini_img_path: str,
    level: str,
    mag: str,
    diagnosis_ref: str,
    inference_summary: str,
    client: OpenAI,
    sem: asyncio.Semaphore
) -> Dict[str, Any]:
    try:
        prompt = get_multilevel_caption_prompt(level, mag, diagnosis_ref, inference_summary)
        img_url = image_to_data_url(gemini_img_path)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]}
        ]
        raw_output = await gpt_call_safe(messages, client, sem, output_json=False)
        captions = parse_caption_abc(raw_output)
        captions["raw_model_output"] = raw_output
        return captions
    except Exception as e:
        print(f"⚠️ Caption failed for {image_id}: {e}")
        return {"caption_a": "", "caption_b": "", "caption_c": "", "error": str(e)}

# ------------------------------------------------------------
# 主函数：为整个 session 生成 captions.jsonl
# ------------------------------------------------------------
async def generate_captions_for_session(session_dir: str,case_json_path: str,):
    client = OpenAI(api_key=API_KEY, base_url=API_URL)  
    sem = asyncio.Semaphore(5)  # 控制并发

    features_dir = os.path.join(session_dir, "features")
    if not os.path.exists(features_dir):
        print("🔹 No features dir. Skipping.")
        return

    inf_path = os.path.join(session_dir, "inference_results.json")
    if not os.path.exists(inf_path):
        raise FileNotFoundError(f"inference_results.json not found in {session_dir}")
    
    with open(inf_path, 'r', encoding='utf-8') as f:
        inf_res = json.load(f)

    wsi_svs = inf_res.get("wsi_svs")
    if not wsi_svs:
        raise ValueError("wsi_svs not found in inference_results.json")
    
    case_meta = get_case_metadata(os.path.join(session_dir, wsi_svs), case_json_path)
    diagnosis_ref = case_meta.get("diagnosis", "N/A")

    def to_rel(p): return os.path.relpath(p, session_dir)

    all_entries: List[Dict] = []
    pt_files = [f for f in os.listdir(features_dir) if f.endswith('.pt')]

    for pt_file in pt_files:
        base = os.path.splitext(pt_file)[0]
        pt_abs = os.path.join(features_dir, pt_file)
        pt_rel = to_rel(pt_abs)

        # --- Determine level, mag, paths ---
        if base == inf_res.get("wsi_id"):
            level, mag = "WSI", "Macro"
            gemini_img = os.path.join(session_dir, "wsi_gemini.png")
            thumb = inf_res.get("wsi_thumbnail_path", "")
            summary = inf_res.get("wsi_summary", "")
            cluster_id, roi_mag = None, None
        elif "_10x" in base or "_20x" in base:
            level = "Patch"
            if "_10x" in base:
                mag, roi_mag = "10x", 10
            else:
                mag, roi_mag = "20x", 20
            cluster_id = base.rsplit('_', 1)[0]
            gemini_img = os.path.join(session_dir, "rois", f"{base}.png")
            thumb = inf_res.get(f"{cluster_id}_roi_{mag}_thumbnail_path", "")
            summary = inf_res.get(f"{cluster_id}_roi_{mag}_summary", "")
        else:
            level, mag = "Cluster", "Medium"
            cluster_id = base
            gemini_img = os.path.join(session_dir, "clusters", f"{base}_gemini.png")
            thumb = inf_res.get(f"{base}_thumbnail_path", "")
            summary = inf_res.get(f"{base}_summary", "")
            roi_mag = None

        if not os.path.exists(gemini_img):
            print(f"⚠️ Missing gemini image: {gemini_img}")
            continue

        print(f"🔹 Processing {level} {base} ({mag})")
        caps = await generate_caption_for_image(
            image_id=base,
            gemini_img_path=gemini_img,
            level=level,
            mag=mag,
            diagnosis_ref=diagnosis_ref,
            inference_summary=summary,
            client=client,
            sem=sem
        )

        subtype_map = [("caption_a", "morphology"), ("caption_b", "microenvironment"), ("caption_c", "clinicopathological")]
        for cap_key, subtype in subtype_map:
            text = caps.get(cap_key, "").strip()
            if not text:
                continue
            entry = {
                "caption_id": generate_caption_id(),
                "image_path": pt_rel,
                "image_thumb_path": thumb,
                "belong_level": level,
                "belong_cluster_id": cluster_id,
                "belong_roi_mag": roi_mag,
                "format": "caption",
                "subtype": subtype,
                "caption": text
            }
            all_entries.append(entry)

    # Save
    out_path = os.path.join(session_dir, "captions.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for ent in all_entries:
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(all_entries)} caption entries to {out_path}")

# ------------------------------------------------------------
# CLI 入口（可选）
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--case_json", required=True)
    args = parser.parse_args()

    asyncio.run(generate_captions_for_session(
        session_dir=args.session_dir,
        case_json_path=args.case_json
    ))