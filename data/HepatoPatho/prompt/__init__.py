# Prompt generation module for pathology analysis
from .prompt import (
    get_infer_prompt, 
    get_qa_prompt, 
    get_captionqa_prompt,
    get_cluster_infer_prompt,
    get_wsi_summary_prompt
)

__all__ = [
    'get_infer_prompt', 
    'get_qa_prompt', 
    'get_captionqa_prompt',
    'get_cluster_infer_prompt',
    'get_wsi_summary_prompt'
]

