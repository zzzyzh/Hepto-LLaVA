#!/usr/bin/env python3
"""
VQA Evaluation Script

Reads VQA data from qa_merged.jsonl, performs inference, and evaluates using metrics.
Compatible with both original WSI-LLaVA and custom connector models.
"""

import argparse
import torch
import os
import json
import sys
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party"))  # enables: import llava, import conch

# Install projector patch BEFORE importing llava modules
# This enables support for custom connector models
from train_wsi_llava_v2 import install_projector_patch
install_projector_patch()

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from utils.metric import exact_match_choice, rouge_l, meteor, compute_metrics


def load_vqa_data(jsonl_path):
    """Load VQA data from qa_merged.jsonl format."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def convert_to_inference_format(sample):
    """Convert qa_merged.jsonl format to inference format."""
    conversations = sample.get("conversations", [])
    
    # Extract question from human message
    question = ""
    if conversations and len(conversations) > 0:
        human_msg = conversations[0].get("value", "")
        # Remove <image> token
        question = human_msg.replace("<image>", "").strip()
    
    # Extract answer from gpt message
    answer = ""
    if len(conversations) > 1:
        answer = conversations[1].get("value", "")
    
    return {
        "question_id": sample.get("id", ""),
        "image": sample.get("image", ""),
        "question": question,
        "answer": answer,
        "info": sample.get("info", {})
    }


def determine_task_type(info):
    """Determine task type (choice or open_ended) from info."""
    format_type = info.get("format", "").lower()
    if format_type in ["single", "multi"]:
        return "choice"
    elif format_type in ["open", "open-ended", "freeform"]:
        return "open_ended"
    
    # Fallback: check if there are options
    if "options" in info or "choices" in info:
        return "choice"
    
    return "open_ended"


def load_image(image_path):
    """
    Load image tensor from .pt file.
    
    Supports two formats:
    1. Direct tensor (legacy format)
    2. Dictionary with 'features' key (new format from feature.py)
    """
    data = torch.load(image_path, map_location='cpu')
    
    # Check if it's a dictionary (new format)
    if isinstance(data, dict):
        if 'features' in data:
            return data['features']
        else:
            raise ValueError(f"Dictionary format but missing 'features' key. Keys: {list(data.keys())}")
    
    # Legacy format: direct tensor
    return data


def run_inference(model, tokenizer, image_processor, questions, args):
    """Run inference on questions."""
    results = []
    
    # Check existing results
    processed_ids = set()
    if os.path.exists(args.output_file):
        print(f"Loading existing results from {args.output_file}")
        with open(args.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed_ids.add(data["question_id"])
                        results.append(data)
                    except:
                        continue
        print(f"Found {len(processed_ids)} existing results, will skip them.")
    
    # Open output file in append mode
    output_file = open(args.output_file, 'a')
    
    for sample in tqdm(questions, desc="Running inference"):
        question_id = sample["question_id"]
        
        # Skip if already processed
        if question_id in processed_ids:
            continue
        
        image_file = sample["image"]
        question = sample["question"]
        ground_truth = sample["answer"]
        info = sample["info"]
        
        # Prepare prompt
        qs = question
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # Load image
        image_path = os.path.join(args.image_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            image = load_image(image_path)
            image_tensor = image.to(model.device, dtype=torch.float16)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Generate
        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size] if hasattr(image, 'size') else None,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    no_repeat_ngram_size=3,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            except Exception as e:
                print(f"Error during generation for {question_id}: {e}")
                continue
        
        prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Save result
        result = {
            "question_id": question_id,
            "image": image_file,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "info": info
        }
        
        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        output_file.flush()
        results.append(result)
    
    output_file.close()
    return results


def evaluate_results(results, output_dir):
    """Evaluate results using metrics from metric.py."""
    
    # Group by task type
    choice_samples = []
    open_ended_samples = []
    
    for result in results:
        task_type = determine_task_type(result.get("info", {}))
        if task_type == "choice":
            choice_samples.append(result)
        else:
            open_ended_samples.append(result)
    
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    all_metrics = {}
    
    # Evaluate choice questions
    if choice_samples:
        print(f"\n[Multiple Choice Questions] N={len(choice_samples)}")
        predictions = [s["prediction"] for s in choice_samples]
        references = [s["ground_truth"] for s in choice_samples]
        
        choice_metrics = compute_metrics(predictions, references, task_type="choice")
        all_metrics["choice"] = choice_metrics
        
        print(f"  Accuracy: {choice_metrics['accuracy']:.4f}")
        print(f"  Correct: {choice_metrics['num_correct']} / {choice_metrics['num_total']}")
        
        # Save detailed choice results
        choice_details = []
        for i, sample in enumerate(choice_samples):
            pred_result = exact_match_choice(
                sample["prediction"], 
                sample["ground_truth"]
            )
            choice_details.append({
                **sample,
                "exact_match": pred_result["exact_match"],
                "pred_letters": pred_result["pred_letters"],
                "ref_letters": pred_result["ref_letters"]
            })
        
        choice_file = Path(output_dir) / "choice_detailed.jsonl"
        with open(choice_file, 'w') as f:
            for detail in choice_details:
                f.write(json.dumps(detail, ensure_ascii=False) + '\n')
        print(f"  Detailed results saved to: {choice_file}")
    
    # Evaluate open-ended questions
    if open_ended_samples:
        print(f"\n[Open-Ended Questions] N={len(open_ended_samples)}")
        predictions = [s["prediction"] for s in open_ended_samples]
        references = [s["ground_truth"] for s in open_ended_samples]
        
        open_metrics = compute_metrics(predictions, references, task_type="open_ended")
        all_metrics["open_ended"] = open_metrics
        
        print(f"  ROUGE-L F1: {open_metrics['rouge_l_fmeasure']:.4f}")
        print(f"  ROUGE-L Precision: {open_metrics['rouge_l_precision']:.4f}")
        print(f"  ROUGE-L Recall: {open_metrics['rouge_l_recall']:.4f}")
        print(f"  METEOR: {open_metrics['meteor_score']:.4f}")
        
        # Save detailed open-ended results
        open_details = []
        for i, sample in enumerate(open_ended_samples):
            rouge_result = rouge_l(sample["prediction"], sample["ground_truth"])
            meteor_result = meteor(sample["prediction"], sample["ground_truth"])
            open_details.append({
                **sample,
                "rouge_l_fmeasure": rouge_result["rouge_l_fmeasure"],
                "meteor_score": meteor_result["meteor_score"]
            })
        
        open_file = Path(output_dir) / "open_ended_detailed.jsonl"
        with open(open_file, 'w') as f:
            for detail in open_details:
                f.write(json.dumps(detail, ensure_ascii=False) + '\n')
        print(f"  Detailed results saved to: {open_file}")
    
    # Overall statistics
    print(f"\n[Overall Statistics]")
    print(f"  Total samples: {len(results)}")
    print(f"  Choice questions: {len(choice_samples)}")
    print(f"  Open-ended questions: {len(open_ended_samples)}")
    
    # Save summary metrics
    summary_file = Path(output_dir) / "metrics_summary.json"
    with open(summary_file, 'w') as f:
        summary = {
            "total_samples": len(results),
            "choice_samples": len(choice_samples),
            "open_ended_samples": len(open_ended_samples),
            "metrics": all_metrics
        }
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nMetrics summary saved to: {summary_file}")
    print("="*60)
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="VQA Evaluation Script")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path for LoRA mode")
    parser.add_argument("--conv-mode", type=str, default="llava_v1",
                       help="Conversation mode")
    
    # Data arguments
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to qa_merged.jsonl file")
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Root folder containing image features")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save results")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p sampling")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    
    # Custom connector arguments (for models with feature padding)
    parser.add_argument("--actual-mm-hidden-size", type=int, default=None,
                       help="Actual dimension of input features (e.g., 512). "
                            "Leave empty for original WSI-LLaVA models.")
    parser.add_argument("--enable-feature-padding", action="store_true", default=True,
                       help="Enable zero-padding for feature dimension mismatch")
    
    # Other arguments
    parser.add_argument("--skip-inference", action="store_true",
                       help="Skip inference and only evaluate existing results")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_file = str(output_dir / "inference_results.jsonl")
    
    print("="*60)
    print("VQA Evaluation")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load data
    print("\nLoading VQA data...")
    vqa_data = load_vqa_data(args.data_file)
    print(f"Loaded {len(vqa_data)} samples")
    
    # Convert to inference format
    questions = [convert_to_inference_format(sample) for sample in vqa_data]
    
    # Run inference or load existing results
    if not args.skip_inference:
        print("\nInitializing model...")
        disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        
        # Treat empty model_base as None (full fine-tune mode)
        if args.model_base is not None and args.model_base.strip() == "":
            args.model_base = None
            print("  [Note] Empty model_base detected, treating as None (full fine-tune mode)")
        
        # If model_base is specified (LoRA mode), ensure model_name contains "llava" and "lora"
        # to trigger correct LLaVA LoRA loading path in builder.py
        # (loads LlavaLlamaForCausalLM + non_lora_trainables.bin + LoRA weights)
        if args.model_base is not None:
            name_lower = model_name.lower()
            if 'llava' not in name_lower or 'lora' not in name_lower:
                original_name = model_name
                model_name = "llava-lora-" + model_name
                print(f"  [Note] Adjusted model_name: '{original_name}' -> '{model_name}' "
                      f"(to trigger LLaVA LoRA loading path)")
        else:
            # Full fine-tune mode: ensure model_name contains "llava"
            # so builder.py uses LlavaLlamaForCausalLM instead of AutoModelForCausalLM
            if 'llava' not in model_name.lower():
                original_name = model_name
                model_name = "llava-" + model_name
                print(f"  [Note] Adjusted model_name: '{original_name}' -> '{model_name}' "
                      f"(to trigger LLaVA full model loading path)")
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name
        )
        
        # If actual_mm_hidden_size is specified, manually set to model config and rebuild projector
        if args.actual_mm_hidden_size is not None:
            model.config.actual_mm_hidden_size = args.actual_mm_hidden_size
            model.config.enable_feature_padding = args.enable_feature_padding
            print(f"[Eval Config] Set actual_mm_hidden_size to {args.actual_mm_hidden_size}")
            
            # Rebuild mm_projector to apply padding
            from train_wsi_llava_v2 import build_custom_projector
            print("[Eval Config] Rebuilding mm_projector with padding...")
            new_projector = build_custom_projector(model.config)
            new_projector = new_projector.to(model.device, dtype=model.dtype)
            model.get_model().mm_projector = new_projector
            print("[Eval Config] mm_projector rebuilt successfully with feature padding enabled")
        
        print("Model loaded successfully")
        
        # Print model config info
        print(f"  mm_hidden_size: {getattr(model.config, 'mm_hidden_size', 'N/A')}")
        print(f"  mm_projector_type: {getattr(model.config, 'mm_projector_type', 'N/A')}")
        if hasattr(model.config, 'mm_projector_path') and model.config.mm_projector_path:
            print(f"  mm_projector_path: {model.config.mm_projector_path}")
        if hasattr(model.config, 'actual_mm_hidden_size') and model.config.actual_mm_hidden_size:
            print(f"  actual_mm_hidden_size: {model.config.actual_mm_hidden_size}")
        
        print("\nRunning inference...")
        results = run_inference(model, tokenizer, image_processor, questions, args)
        print(f"Inference completed: {len(results)} samples")
    else:
        print("\nSkipping inference, loading existing results...")
        if not os.path.exists(args.output_file):
            print(f"Error: Results file not found: {args.output_file}")
            return
        
        results = []
        with open(args.output_file, 'r') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        print(f"Loaded {len(results)} existing results")
    
    # Evaluate results
    if results:
        print("\nEvaluating results...")
        metrics = evaluate_results(results, output_dir)
    else:
        print("\nNo results to evaluate")


if __name__ == "__main__":
    main()
