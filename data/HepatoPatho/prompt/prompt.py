#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random

# 加载提示词JSON文件（相对于此文件的路径）
_PROMPT_JSON_PATH = os.path.join(os.path.dirname(__file__), "HCC_diagnostic_prompts.json")
# Knowledge JSON filenames (will be created if missing)
HCC_JSON = "./hcc_expert_knowledge.json"
FIBROSIS_JSON = "./fibrosis_cirrhosis_knowledge.json"
ICCA_JSON = "./cholangiocarcinoma_knowledge.json"

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
# Load knowledge globals
HCC_KNOWLEDGE = load_json(HCC_JSON)
FIBROSIS_KNOWLEDGE = load_json(FIBROSIS_JSON)
ICCA_KNOWLEDGE = load_json(ICCA_JSON)

def get_infer_prompt(mag, diagnosis_ref=""):
    """生成ROI推理提示词"""
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
    combined_prompt = f"""
You are a professional hepatopathologist specializing in liver tumors.

Reference diagnosis (for pathologist only): {diagnosis_ref}

{know_block}

Please analyze the provided histopathological ROI at {view_desc}.

Microscopic diagnostic focus:
{micro_prompt}

Complementary diagnostic considerations (reference only):
1. {major_prompts[0]}
2. {major_prompts[1]}
3. {major_prompts[2]}
4. {major_prompts[3]}

Please structure the response:
1. Tissue Quality & Composition
2. Architectural Pattern
3. Cytological Features
4. Differential Diagnosis
5. Diagnostic Reasoning

Length: 300–450 words.
Strictly based on what is VISIBLE in the ROI.
"""

    return combined_prompt

def get_qa_prompt(diagnosis_text, inference_summary, question_type, difficulty):
    if "multiple" in question_type.lower():
        extra_rule = "This is a MULTIPLE-CHOICE question. At least TWO correct answers."
    else:
        extra_rule = "This is a SINGLE-CHOICE question. EXACTLY ONE correct answer."
    
    difficulty_text = f"The difficulty of this question MUST be **{difficulty.upper()}**."

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

**Task:**
Generate ONE {question_type} question.
{difficulty_text}

Return JSON in this structure (MUST include 'difficulty' key):
{{
  "question": "....",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "answer": ["A"] or ["A", "C"],
  "explanation": {{
    "A": "...", "B": "...", "C": "...", "D": "..."
  }},
  "difficulty": "{difficulty}" 
}}

{extra_rule}
"""

def get_captionqa_prompt(diagnosis_text, inference_summary, magnification):
    focus_options = {
        10: ["architecture","stroma","inflammation"],
        20: ["cytology","architecture","necrosis"],
        40: ["cytology","nuclear_detail","stroma"]
    }.get(magnification, ["diagnosis"])
    return f"""
You are a professional hepatopathologist.

**Visual Context (Visual Findings):**
{inference_summary}

**Ground Truth Diagnosis:**
{diagnosis_text}

=== Expert Knowledge Constraints ===
=== HCC Diagnostic Knowledge ===
{json.dumps(HCC_KNOWLEDGE, indent=2)}

=== Liver Fibrosis & Cirrhosis Knowledge ===
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

=== Intrahepatic Cholangiocarcinoma Knowledge ===
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

**Task:**
Observe the provided liver histopathology ROI image at {magnification}× magnification.
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

def get_cluster_infer_prompt(cluster_id, roi_texts, diagnosis):
    """生成Cluster级别推理提示词"""
    combined = "\n\n".join(roi_texts) if roi_texts else "(No ROI textual context available.)"
    prompt = prompt = f"""
You are a senior hepatopathologist.

Reference diagnosis: {diagnosis}

=== Expert Hepatopathology Knowledge ===
--- HCC Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Cirrhosis/Fibrosis Background ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA vs HCC Differentiation ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

This task: Analyze the provided CLUSTER-LEVEL image for Cluster {cluster_id}.

Use the ROI-level inference context below (these are micro findings from 10x/20x/40x ROIs extracted from this cluster):

{combined}

Now integrate:
1. ROI micro findings
2. Cluster-level image features
3. Expert hepatopathology diagnostic knowledge

Output a 300–600 word cluster-level summary, describing:
- Architecture
- Cytology
- Stroma
- Necrosis
- Vascular/bile duct changes
- Impression (non-diagnostic, descriptive)

Do **not** mention the diagnosis text directly.
"""
    return prompt

def get_wsi_summary_prompt(cluster_summaries, diagnosis):
    """生成WSI级别总结提示词"""
    cluster_text = "\n\n".join([f"Cluster {c['cluster_id']} Summary:\n{c['summary']}" for c in cluster_summaries])
    prompt = f"""
You are a senior hepatopathologist.

Reference diagnosis (for your understanding only): {diagnosis}

=== Expert Global Knowledge ===
--- HCC Knowledge ---
{json.dumps(HCC_KNOWLEDGE, indent=2)}

--- Liver Fibrosis / Cirrhosis ---
{json.dumps(FIBROSIS_KNOWLEDGE, indent=2)}

--- iCCA Knowledge ---
{json.dumps(ICCA_KNOWLEDGE, indent=2)}

Task:
Integrate the cluster-level summaries with the global architectural pattern visible in the thumbnail image.

=== Cluster-level summaries provided ===
{cluster_text}

Describe:
1. Global tissue distribution
2. Relationship between tumor & cirrhotic background
3. Architectural patterns
4. Suspicion level (descriptive, not committing to diagnosis)

Length: 300–500 words.
Do not restate the diagnosis.
"""
    return prompt

