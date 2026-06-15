#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random

# 加载提示词JSON文件（相对于此文件的路径）
_PROMPT_JSON_PATH = os.path.join(os.path.dirname(__file__), "HCC_diagnostic_prompts.json")
_HCC_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "hcc_expert_knowledge.json")
_FIBROSIS_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "fibrosis_cirrhosis_knowledge.json")
_ICCA_KNOWLEDGE_JSON_PATH = os.path.join(os.path.dirname(__file__), "cholangiocarcinoma_knowledge.json")

def _load_prompts():
    """加载提示词数据"""
    if os.path.exists(_PROMPT_JSON_PATH):
        with open(_PROMPT_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_json(p):
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# 在模块加载时读取提示词数据
_PROMPTS_DATA = _load_prompts()
HCC_KNOWLEDGE = load_json(_HCC_KNOWLEDGE_JSON_PATH) 
FIBROSIS_KNOWLEDGE = load_json(_FIBROSIS_KNOWLEDGE_JSON_PATH)
ICCA_KNOWLEDGE = load_json(_ICCA_KNOWLEDGE_JSON_PATH) 

def get_infer_prompt(mag, diagnosis_ref="", cluster_context="", wsi_context=""): 
    
    # ---- Inject Medical Knowledge ----
    know_block = f"""
=== Hepatopathology Expert Knowledge ===

--- HCC Diagnostic Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Liver Fibrosis / Cirrhosis Knowledge ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA (Cholangiocarcinoma) Knowledge ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}
"""
    
    if not _PROMPTS_DATA or "HCC_diagnostic_prompts" not in _PROMPTS_DATA:
          return f"Analyze this {mag}x pathology image based on diagnosis: {diagnosis_ref}."
    micro_prompts = _PROMPTS_DATA["HCC_diagnostic_prompts"]["2_Microscopic_Diagnosis"]
    if mag == 10:
        view_prompts = micro_prompts["Low_Power_View"]
        view_desc = "low-power view (10x)"
    elif mag == 20:
        view_prompts = micro_prompts["Medium_Power_View"]
        view_desc = "medium-power view (20x)"
    else:
        view_prompts = micro_prompts["High_Power_View"]
        view_desc = f"high-power view ({mag}x)"
    micro_prompt = random.choice(view_prompts)
    major_prompts = []
    keys = ["1_Specimen_Gross_Evaluation","3_Immunohistochemistry","4_Differential_Diagnosis","5_Final_Diagnosis"]
    for key in keys:
        major_prompts.append(random.choice(_PROMPTS_DATA["HCC_diagnostic_prompts"][key]))

    context_block = ""
    if wsi_context:
        # 完整传递 WSI 上下文
        context_block += f"\n--- WSI GLOBAL CONTEXT ---\n{wsi_context}\n" 
    if cluster_context:
        # 完整传递 Cluster 上下文
        context_block += f"\n--- CLUSTER LOCAL CONTEXT ---\n{cluster_context}\n"

    combined_prompt = f"""
You are a professional hepatopathologist specializing in liver tumors.

Reference diagnosis (for pathologist only): {diagnosis_ref}

{know_block}

{context_block}

Please analyze the provided histopathological ROI at {view_desc}.

Microscopic diagnostic focus:
{micro_prompt}

Complementary diagnostic considerations (reference only):
1. {major_prompts[0]}
2. {major_prompts[1]}
3. {major_prompts[2]}
4. {major_prompts[3]}

Please structure the response:
1. Tissue Quality & Composition (Focus on {mag}x detail)
2. Architectural Pattern
3. Cytological Features
4. Differential Diagnosis
5. Diagnostic Reasoning

Length: 300–450 words.
Strictly based on what is VISIBLE in the ROI. Integrate the WSI and Cluster context to enhance your micro-level analysis.
"""

    return combined_prompt

def get_cluster_infer_prompt(cluster_id, wsi_context, diagnosis):
    prompt = f"""
You are a senior hepatopathologist.

Reference diagnosis: {diagnosis}

=== Expert Hepatopathology Knowledge ===
--- HCC Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Cirrhosis/Fibrosis Background ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA vs HCC Differentiation ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

**WSI Global Context (From Thumbnail Analysis):**
{wsi_context}

This task: Analyze the provided CLUSTER-LEVEL image for Cluster {cluster_id}.

Integrate:
1. WSI macro context
2. Cluster-level image features (architecture, cell size, density)
3. Expert hepatopathology diagnostic knowledge

Output a 300–600 word cluster-level summary, describing:
- Architecture (in relation to WSI context)
- Cytology (initial impression)
- Stroma/Background tissue changes
- Impression (non-diagnostic, descriptive)

Do **not** mention the diagnosis text directly.
"""
    return prompt

def get_wsi_infer_prompt(diagnosis_ref):
    prompt = f"""
You are a senior hepatopathologist.

Reference diagnosis (for your understanding only): {diagnosis_ref}

=== Expert Global Knowledge ===
--- HCC Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Liver Fibrosis / Cirrhosis ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA Knowledge ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

Task:
Analyze the provided WSI thumbnail image. This represents the global, macro-architectural context of the liver tissue.

Describe:
1. Global tissue distribution (e.g., solid mass, multifocal, cirrhotic background)
2. Overall architectural pattern (e.g., nodular, diffuse)
3. Relationship between different tissue components (e.g., tumor vs non-tumor)
4. Key macroscopic findings (e.g., necrosis, hemorrhage, large vessels)

Length: 300–500 words.
Do not restate the diagnosis.
"""
    return prompt

# ============================================================
# VQA 任务分类映射 (基于 wsi-bench)
# ============================================================
VQA_CATEGORIES = {
    "1.1": "Global Morphology Description",
    "1.2": "Key Diagnostic Description",
    "1.3": "Regional Structure Description",
    "1.4": "Specific Feature Description",
    "2.1": "Histological Typing",
    "2.2": "Grading",
    "2.3": "Molecular Subtyping",
    "2.4": "Staging"
}

def get_vqa_prompt(category_code, sub_type, metadata, inference_summary, magnification):
    """
    生成符合 wsi-bench 逻辑的 VQA Prompt
    :param category_code: "1.1" - "2.4"
    :param sub_type: "single", "multi", "caption"
    :param metadata: case_info_json 中的单个病例字典
    :param inference_summary: LLM 对图像的初步形态推理
    :param magnification: 放大倍率
    """
    
    cat_name = VQA_CATEGORIES.get(category_code, "General Diagnosis")
    is_morphology = category_code.startswith("1")
    
    # 1. 设定系统角色与基础准则 
    if is_morphology:
        system_role = "You are a professional hepatopathologist specializing in liver tumor pathology."
        core_rule = "OBSERVATION MODE: Directly viewing the slide. Describe morphological features. DO NOT mention diagnosis, prognosis, or grading."
    else:
        system_role = "You are a professional hepatopathologist specializing in liver tumor pathology."
        core_rule = "DIAGNOSTIC MODE: Use visual evidence and ground truth to conclude pathology classification, grading, or staging."

    # 2. 准备子类特定的知识和 Ground Truth
    gt_diagnosis = metadata.get("diagnosis", "N/A")
    
    task_specific_instr = ""
    if category_code == "1.1": # Global
        task_specific_instr = f"Focus on overall tissue distribution, borders (capsule), and size. Ground Truth: {gt_diagnosis}"
    elif category_code == "1.2": # Key Features
        task_specific_instr = f"Focus on necrosis, hemorrhage, satellite nodules, or vessel invasion.Ground Truth: {gt_diagnosis}"
    elif category_code == "1.3": # Regional
        task_specific_instr = f"Focus on tumor infiltration, architectural patterns (trabecular/solid), and cell density.Ground Truth: {gt_diagnosis}"
    elif category_code == "1.4": # Specific
        task_specific_instr = f"Focus on high-power details: nuclear atypia, mitoses, Mallory bodies, or bile droplets.Ground Truth: {gt_diagnosis}"
    elif category_code == "2.1": # Typing
        task_specific_instr = f"Determine the histological type. Ground Truth: {gt_diagnosis}"
    elif category_code == "2.2": # Grading
        task_specific_instr = f"Assign a differentiation grade (e.g., Edmondson). Ground Truth: {gt_diagnosis}"
    elif category_code == "2.3": # Molecular
        task_specific_instr = f"Predict molecular features or IHC results. Ground Truth IHC: {gt_diagnosis}"
    elif category_code == "2.4": # Staging
        task_specific_instr = f"Determine TNM staging based on size and invasion. Ground Truth: {gt_diagnosis}"

    # 3. 设定题型约束 (A-F 六选一/多)
    if sub_type == "single":
        format_rule = "SINGLE-CHOICE: Exactly ONE correct answer. Options: A, B, C, D, E, F. Explanation: A: ..., B: ..., C: ..., D: ..., E: ..., F: ..."
    elif sub_type == "multi":
        format_rule = "MULTIPLE-CHOICE: At least TWO correct answers. Options: A, B, C, D, E, F.Explanation: A: ..., B: ..., C: ..., D: ..., E: ..., F: ..."
    else: # caption
        format_rule = f"""CAPTION-QA: Generate a list of objects with 'caption', 'question', 'answer',  'focus'.
            Each pair must include:
            1. **caption**: Concise histologic description (1-2 sentences).
            2. **question**: A diagnostically meaningful question about the feature described.
            3. **answer**: Detailed educational reasoning.
            5. **focus**: Choose from {category_code}.
        """

    # 4. 组合最终 Prompt
    prompt = f"""
{system_role}
{core_rule}

Task: Generate a {sub_type.upper()} question for Category {category_code} ({cat_name}).

[Visual Morphology Context]:
{inference_summary}

[Pathology Report Context (Ground Truth)]:
{task_specific_instr}

=== Expert Global Knowledge ===
--- HCC Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Liver Fibrosis / Cirrhosis ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA Knowledge ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

=== Output Requirement ===
{format_rule}
All options MUST be professional pathological entities relevant to liver lesions.
Return strictly in JSON format.
"""
    return prompt