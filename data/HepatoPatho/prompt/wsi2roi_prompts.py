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

def get_qa_prompt(diagnosis_text, inference_summary, question_type, difficulty):
    if "multiple" in question_type.lower():
        extra_rule = "This is a MULTIPLE-CHOICE question. At least TWO correct answers."
    else:
        extra_rule = "This is a SINGLE-CHOICE question. EXACTLY ONE correct answer."
    
    difficulty_text = f"The difficulty of this question MUST be **{difficulty.upper()}**."

    option_rules = """
**Crucial Option Generation Constraint :**
1.  **Relevance/Professionalism:** ALL options MUST be highly relevant to the differential diagnosis of LIVER LESIONS, based on the provided expert knowledge and visual context.
    -   Options must be professional pathological entities (e.g., High-grade HCC, Low-grade HCC, FNH, Metastatic Colorectal Carcinoma, Cirrhotic Pseudonodule, Hemangioma, etc.).
    -   NEVER include options describing non-liver lesions (e.g., lung adenocarcinoma, breast fibroadenoma) unless they are known metastases to the liver (e.g., metastatic adenocarcinoma).
2.  **Depth:** Use descriptions that cover the spectrum of liver pathology:
    -   Differentiation grades (High/Low Grade HCC)
    -   Background changes (Cirrhosis, Fibrosis, Steatohepatitis)
    -   Benign lesions (FNH, Adenoma, Haemangioma)
    -   Metastatic tumors (e.g., colon, breast, pancreas metastasis to liver)
"""
    return f"""
You are a pathology education assistant.

**Constraint: Blind Evaluation**
- The student answering this question DOES NOT have access to the pathology report.
- You must NOT generate questions that ask "What does the report say?" or "Does this match the report?".
- You MUST generate questions based on the **visual morphology** visible in the image.
- You CAN ask to **predict** likely Immunohistochemistry (IHC) results or molecular features based on the visual pattern seen (e.g., "Given the trabecular pattern, which stain is likely positive?").

**Context (Ground Truth for your reference only):**
Final Diagnosis: {diagnosis_text}

**Visual Context (What the model 'sees'):**
{inference_summary}

=== Expert Knowledge Constraints ===
=== HCC Diagnostic Knowledge ===
{json.dumps(HCC_KNOWLEDGE, indent=2)}

=== Liver Fibrosis & Cirrhosis Knowledge ===
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

=== Intrahepatic Cholangiocarcinoma Knowledge ===
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

{option_rules}

**Task:**
Generate ONE {question_type} question.
{difficulty_text}

Return JSON in this structure (MUST include 'difficulty' key):
{{
  "question": "....",
  "options": ["A. ...", "B. ...", "C. ...", "D. ...", "E. ...", "F. ..."], 
  "answer": ["A"] or ["A", "C", "F"],
  "explanation": {{
    "A": "...", "B": "...", "C": "...", "D": "...", "E": "...", "F": "..."
  }},
  "difficulty": "{difficulty}" 
}}

{extra_rule}
"""

def get_captionqa_prompt(diagnosis_text, inference_summary, magnification):
    focus_options = {
        10: ["architecture","stroma","inflammation"],
        20: ["cytology","architecture","necrosis"],
        40: ["cytology","nuclear_detail","stroma"],
        2048: ["macro_architecture", "global_context", "diagnosis"] # For WSI/Cluster
    }.get(magnification, ["diagnosis"])
    
    # 调整对 WSI/Cluster 的放大倍数描述
    if magnification >= 1024:
        mag_desc = "Cluster/WSI Low-Power View"
    else:
        mag_desc = f"{magnification}× magnification"

    return f"""
You are a professional hepatopathologist.

**Visual Context (Visual Findings):**
{inference_summary}

**Ground Truth Diagnosis:**
{diagnosis_text}

=== Expert Knowledge Constraints ===
{json.dumps(HCC_KNOWLEDGE, indent=2)}
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

**Task:**
Observe the provided liver histopathology image at {mag_desc}.
Generate **3 closed-ended caption–QA pairs**.

**Strict Constraints:**
1. Do NOT mention "According to the report".
2. Questions must be answerable by looking at the image (or predicting features based on the image).

Each pair must include:
1. **caption**: Concise histologic description (1-2 sentences).
2. **question**: A diagnostically meaningful question about the feature described.
3. **answer**: Detailed educational reasoning.
4. **difficulty**: "easy", "medium", or "hard".
5. **focus**: Choose from {focus_options}.

Return strictly as a JSON LIST of objects:
[
  {{ "caption": "...", "question": "...", "answer": "...", "difficulty": "...", "focus": "..." }},
  ...
]
"""

def get_cluster_infer_prompt(cluster_id, wsi_context, diagnosis):
    # 保持不变
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