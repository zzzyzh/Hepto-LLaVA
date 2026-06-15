#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random

# 加载JSON文件（相对于此文件的路径）
_HCC_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "hcc_expert_knowledge.json")
_FIBROSIS_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "fibrosis_cirrhosis_knowledge.json")
_ICCA_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "cholangiocarcinoma_knowledge.json")

def load_json(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 在模块加载时读取数据
HCC_KNOWLEDGE = load_json(_HCC_KNOWLEDGE_JSON_PATH) 
FIBROSIS_KNOWLEDGE = load_json(_FIBROSIS_KNOWLEDGE_JSON_PATH)
ICCA_KNOWLEDGE = load_json(_ICCA_KNOWLEDGE_JSON_PATH) 

def get_multilevel_caption_prompt(level, mag, diagnosis_ref, inference_summary=""):
    """
    针对不同层级（WSI, Cluster, Patch）生成差异化 Caption 的 Prompt
    :param level: "WSI", "Cluster", 或 "Patch"
    :param mag: 放大倍率（WSI通常为low, Patch为10x/20x）
    :param diagnosis_ref: 真实诊断信息
    :param inference_summary: 之前的推理结果
    """
    
    # 针对不同层级的描述侧重点定义
    level_focus = {
        "WSI": "Focus on global tissue architecture, lesion distribution, and the relationship between tumor and non-tumor regions.",
        "Cluster": "Focus on localized architectural growth patterns, nodular formations, and the surrounding stromal microenvironment.",
        "Patch": f"Focus on microscopic cytological details, nuclear atypia, and specific diagnostic hallmarks visible at {mag} magnification."
    }
    
    # 注入专家知识库
    know_block = f"""
HCC Knowledge: {json.dumps(HCC_KNOWLEDGE)}
Fibrosis Knowledge: {json.dumps(FIBROSIS_KNOWLEDGE)}
iCCA Knowledge: {json.dumps(ICCA_KNOWLEDGE)}
"""

    prompt = f"""
You are a senior hepatopathologist. Generate 3 distinct captions for a {level}-level liver pathology image.

[Contextual Information]
- Image Level: {level}
- Magnification: {mag if level == 'Patch' else 'Global View'}
- Confirmed Diagnosis: {diagnosis_ref}
- Morphology Summary: {inference_summary}

[Expert Knowledge Base]:

{know_block}

(Usage Strategy):
    1. **Terminology Check**: Map the visual features in "Morphology Summary" to the standard histological terms found in this knowledge base (e.g., if you see "thick plates", use terms like "trabecular thickening" from the HCC entry).
    2. **Feature Validation**: Ensure the described features are biologically consistent with the "Confirmed Diagnosis" using the provided traits.
    3. **No Plagiarism**: Do not simply copy the JSON structure. Weave the concepts naturally into the sentences.
    4. **Clinical Reasoning**: Use the knowledge to explain *why* the visible features support the diagnosis in Caption C.
    5. **Contextual Filtering**: Only reference features that are visibly plausible based on the 'Morphology Summary'. Do not list all knowledge entries; select only what is relevant to this image view.

[Requirements]
1. Generate 3 separate captions (Caption A, Caption B, Caption C).
2. Each caption must be a complete, professional description limited to exactly 4 sentences.
3. NO Markdown formatting (no bold, no asterisks, no headers).
4. Level-Specific Focus: {level_focus.get(level)}

[Differentiation Strategy]
- Caption A (Morphology): Strict description of visible structures in this specific {level} view.
- Caption B (Microenvironment): Describe stroma, inflammation, or texture.
- Caption C (Clinicopathological Correlation): Relate visible features to the diagnosis of {diagnosis_ref}. **If the visual evidence does not support the diagnosis (e.g., normal tissue in a tumor slide), describe what is visible strictly.**

[Strict Generation Rules]
1. **Visual Truth Hierarchy (Image > Diagnosis > Summary)**: 
   - Base descriptions strictly on visible evidence; if the specific view shows benign tissue or stroma, describe it as context-associated non-neoplastic tissue, explicitly overriding the malignant diagnosis label and any conflicting summary details.
   
2. **Professional Style Guide (Negative Constraints)**:
   - NO subjective adjectives (e.g., "chaotic", "scary"). Use standardized grading terms (e.g., "disordered", "pleomorphic").
   - NO flowery language. Keep it dense and clinical.

3. **Anti-Hallucination**:
   - Do not invent features like "bile plugs" or "pseudo-glands" unless they are in the 'Morphology Summary' or strongly implied by the specific visual evidence provided.
   - If the 'Morphology Summary' is sparse, generate a concise caption. Do not pad with generic fluff.

[Output Format]
Caption A: [Full text]
Caption B: [Full text]
Caption C: [Full text]
"""
    return prompt