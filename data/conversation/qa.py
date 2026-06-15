"""
Convert all session vqa.jsonl to LLaVA fine-tuning data format
and move feature files to output directory
"""
import json
import sys
import shutil
import re
from pathlib import Path


def normalize_option_format(option_str):
    """Normalize option string to 'A. ...' format
    
    Supported input formats:
    - "A. content" -> "A. content"
    - "A: content" -> "A. content"
    - "A content" -> "A. content"
    """
    option_str = option_str.strip()
    # Match "A." or "A:" or "A " prefix
    match = re.match(r'^([A-Z])[\.\:\s]+(.*)$', option_str)
    if match:
        letter, content = match.groups()
        return f"{letter}. {content.strip()}"
    return option_str


def format_options(options, option_descriptions=None, choices=None):
    """Format options to string, unified output as 'A. ...' format
    
    Supported formats:
    1. {"A": "...", "B": "..."} - dict format
    2. [{"key": "A", "label": "..."}] - object list format
    3. ["A", "B", "C"] + option_descriptions={"A": "...", "B": "..."} - separate format
    4. ["A. ...", "B. ..."] - string list format (normalize to dot)
    5. ["A: ...", "B: ..."] - colon-separated string list format (normalize to dot)
    6. "A. ..., B. ..., C. ..." - comma-separated string format (normalize to dot)
    
    Note:
    - option_descriptions is compatible with option_content field
    - choices field is equivalent to dict format options
    """
    # If choices field exists, use it first (equivalent to dict format)
    if choices and isinstance(choices, dict):
        return "\n".join(f"{k}. {v}" for k, v in choices.items())
    
    if options is None:
        return ""
    
    # Handle pure string format
    if isinstance(options, str):
        # Format 6: "A. ..., B. ..., C. ..." -> "A. ...\nB. ...\nC. ..."
        parts = [normalize_option_format(part) for part in options.split(',')]
        return "\n".join(parts)
    
    if isinstance(options, dict):
        # Format 1: {"A": "...", "B": "..."} -> "A. ...\nB. ..."
        return "\n".join(f"{k}. {v}" for k, v in options.items())
    
    elif isinstance(options, list):
        if not options:
            return ""
        
        # Check first element type
        if isinstance(options[0], dict):
            # Format 2: [{"key": "A", "label": "..."}, ...] -> "A. ...\nB. ..."
            if 'key' in options[0] and 'label' in options[0]:
                return "\n".join(f"{opt['key']}. {opt['label']}" for opt in options)
            else:
                # Try other possible key names
                return "\n".join(str(opt) for opt in options)
        
        elif isinstance(options[0], str):
            # Check if option_descriptions or option_content exists
            if option_descriptions and isinstance(option_descriptions, dict):
                # Format 3: ["A", "B"] + {"A": "...", "B": "..."} -> "A. ...\nB. ..."
                return "\n".join(f"{key}. {option_descriptions[key]}" for key in options if key in option_descriptions)
            else:
                # Format 4/5: string list - normalize format
                normalized = [normalize_option_format(opt) for opt in options]
                return "\n".join(normalized)
    
    return ""


def index_to_letter(index):
    """Convert index to option letter (0→A, 1→B, 2→C, ...)"""
    if isinstance(index, int):
        return chr(ord('A') + index)
    return str(index)


def convert_answer_index(answer_index):
    """Convert answer_index to option letter
    
    Examples:
        [0, 1, 3] -> ["A", "B", "D"]
        2 -> "C"
    """
    if answer_index is None:
        return None
    
    if isinstance(answer_index, list):
        return [index_to_letter(idx) for idx in answer_index]
    else:
        return index_to_letter(answer_index)


def format_answer(answer):
    """Format answer
    
    Supports answer or answer_key field
    """
    if isinstance(answer, list):
        return ", ".join(str(a) for a in answer)
    return str(answer) if answer else ""


def get_answer_value(item):
    """Get answer value, search and convert by priority
    
    Priority: answer_index(converted) > answer_indices(converted) > answer_idx(converted) >
            answer_key > answer_keys > correct_answers > correct_answer > 
            answer_choice > answer_options > answer
    
    Note: answer_index/answer_indices/answer_idx will convert index to letter
          If these index fields exist, use converted value first, ignore answer field
    """
    # Handle index fields that need conversion first
    answer_index = item.get("answer_index")
    if answer_index is not None:
        return convert_answer_index(answer_index)
    
    answer_indices = item.get("answer_indices")
    if answer_indices is not None:
        return convert_answer_index(answer_indices)
    
    answer_idx = item.get("answer_idx")
    if answer_idx is not None:
        return convert_answer_index(answer_idx)
    
    # Then search other answer fields
    answer = (item.get("answer_key") or 
              item.get("answer_keys") or
              item.get("correct_answers") or 
              item.get("correct_answer") or 
              item.get("answer_choice") or
              item.get("answer_options") or
              item.get("answer"))
    
    return answer


def should_filter_sample(item):
    """Determine if sample should be filtered
    
    Returns: (should_filter, reason)
    """
    # Check if question is empty
    question = item.get("question", "")
    if not question or not question.strip():
        return True, "empty question"
    
    # Check if all possible answer fields are empty
    answer = get_answer_value(item)
    if not answer or (isinstance(answer, list) and len(answer) == 0):
        return True, "empty answer"
    
    # Check options for single/multi format
    format_type = item.get("format", "")
    if format_type in ["single", "multi"]:
        options = item.get("options")
        option_descriptions = item.get("option_descriptions") or item.get("option_content")
        choices = item.get("choices")
        
        # Get concatenated options string
        try:
            options_str = format_options(options, option_descriptions, choices)
        except Exception as e:
            # If formatting fails, don't filter (decide in later processing)
            return False, ""
        
        if len(options_str) < 10:
            return True, f"format={format_type} and options length<10 (actual={len(options_str)})"
    
    return False, ""


def process_direct_format(input_path, output_path, features_dir, samples, 
                          filtered_records, file_mapping, stats):
    """Process new format: directory structure with vqa.jsonl and features directory"""
    jsonl_file = input_path / "vqa.jsonl"
    input_features_dir = input_path / "features"
    
    # Read jsonl
    with open(jsonl_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            stats["total"] += 1
            
            # Check if sample should be filtered
            should_filter, filter_reason = should_filter_sample(item)
            if should_filter:
                item_id = item.get("id", item.get("qa_id", "unknown"))
                print(f"⚠️  Filtered: {item_id} - Reason: {filter_reason}")
                filtered_records.append({
                    "original_record": item,
                    "filter_reason": filter_reason
                })
                stats["filtered"] += 1
                continue
            
            # Process image path
            image_path = item.get("image", "")
            if image_path and image_path.endswith('.pt'):
                # Extract filename if path starts with features/
                if image_path.startswith("features/"):
                    filename = Path(image_path).name
                else:
                    filename = Path(image_path).name
                
                src_file = input_features_dir / filename
                
                if src_file.exists():
                    # Copy feature file to output directory
                    dst_file = features_dir / filename
                    if not dst_file.exists():
                        shutil.copy2(src_file, dst_file)
                    new_image_path = f"features/{filename}"
                else:
                    item_id = item.get("id", item.get("qa_id", "unknown"))
                    print(f"⚠️  Filtered: {item_id} - Reason: feature file not found {src_file}")
                    filtered_records.append({
                        "original_record": item,
                        "filter_reason": f"feature file not found: {src_file}"
                    })
                    stats["filtered"] += 1
                    continue
            else:
                new_image_path = image_path
            
            # Build human content
            question = item.get("question", "")
            option_desc = item.get("option_descriptions") or item.get("option_content")
            choices = item.get("choices")
            options_str = format_options(
                item.get("options"), 
                option_desc,
                choices
            )
            human_content = f"<image>\n{question}"
            if options_str:
                human_content += f"\n{options_str}"
            
            # Build gpt content
            answer_value = get_answer_value(item) or ""
            gpt_content = format_answer(answer_value)
            
            # Build info
            info = {}
            for key in ["belong_level", "belong_cluster_id", "belong_roi_mag", 
                        "focus", "subcategory", "format", "option_descriptions", 
                        "option_content", "choices", "explanation", "answer_key", 
                        "answer_keys", "correct_answers", "correct_answer", 
                        "answer_indices", "answer_index", "answer_choice", 
                        "answer_idx", "answer_options"]:
                if key in item:
                    info[key] = item[key]
            
            sample = {
                "id": item.get("id", item.get("qa_id", "")),
                "image": new_image_path,
                "conversations": [
                    {"from": "human", "value": human_content},
                    {"from": "gpt", "value": gpt_content}
                ],
                "info": info
            }
            samples.append(sample)
            stats["success"] += 1
    
    print(f"Processing complete, records: {stats['success']}")
    
    # Save results
    save_results(output_path, samples, filtered_records, stats, features_dir)


def batch_convert_sessions(input_dir: str, output_dir: str):
    """Batch process all session directories with vqa.jsonl or directly process merged vqa.jsonl"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create features directory
    features_dir = output_path / "features"
    features_dir.mkdir(exist_ok=True)
    
    samples = []
    filtered_records = []  # Store filtered original records
    file_mapping = {}  # Record mapping from original path to new path
    stats = {"total": 0, "filtered": 0, "success": 0}  # Statistics
    
    # Detect directory structure: if directly contains vqa.jsonl, it's new format
    direct_jsonl = input_path / "vqa.jsonl"
    if direct_jsonl.exists():
        print(f"Detected new format directory structure (directly contains vqa.jsonl)")
        process_direct_format(input_path, output_path, features_dir, samples, 
                            filtered_records, file_mapping, stats)
        return
    
    # Otherwise use old format: collect all sessions
    sessions = sorted(input_path.glob("session_*"))
    print(f"Detected old format directory structure, found {len(sessions)} session directories")
    
    for session_dir in sessions:
        jsonl_file = session_dir / "vqa.jsonl"
        if not jsonl_file.exists():
            print(f"Skip {session_dir.name}: vqa.jsonl not found")
            continue
        
        session_name = session_dir.name
        session_count = 0
        
        # Read jsonl
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                stats["total"] += 1
                
                # Check if sample should be filtered
                should_filter, filter_reason = should_filter_sample(item)
                if should_filter:
                    qa_id = item.get("qa_id", "unknown")
                    print(f"⚠️  Filtered: {session_dir.name}/{qa_id} - Reason: {filter_reason}")
                    filtered_records.append({
                        "session": session_name,
                        "original_record": item,
                        "filter_reason": filter_reason
                    })
                    stats["filtered"] += 1
                    continue
                
                # Process image_path
                new_image_path = item.get("image_path", "")
                if new_image_path.endswith('.pt'):
                    old_image_path = new_image_path
                    src_file = session_dir / old_image_path
                    file_key = f"{session_name}/{old_image_path}"
                    
                    if file_key not in file_mapping:
                        if src_file.exists():
                            # New filename: session_originalfilename
                            filename = Path(old_image_path).name
                            new_filename = f"{session_name}_{filename}"
                            dst_file = features_dir / new_filename
                            shutil.copy2(src_file, dst_file)
                            file_mapping[file_key] = f"features/{new_filename}"
                        else:
                            qa_id = item.get("qa_id", "unknown")
                            print(f"⚠️  Filtered: {session_dir.name}/{qa_id} - Reason: feature file not found {src_file}")
                            filtered_records.append({
                                "session": session_name,
                                "original_record": item,
                                "filter_reason": f"feature file not found: {src_file}"
                            })
                            stats["filtered"] += 1
                            continue
                    
                    new_image_path = file_mapping[file_key]
                
                # Build human content
                question = item.get("question", "")
                # Compatible with option_descriptions, option_content and choices
                option_desc = item.get("option_descriptions") or item.get("option_content")
                choices = item.get("choices")
                options_str = format_options(
                    item.get("options"), 
                    option_desc,
                    choices
                )
                human_content = f"<image>\n{question}"
                if options_str:
                    human_content += f"\n{options_str}"
                
                # Build gpt content (get answer field by priority, auto-convert answer_index)
                answer_value = get_answer_value(item) or ""
                gpt_content = format_answer(answer_value)
                
                # Build info
                info = {}
                for key in ["belong_level", "belong_cluster_id", "belong_roi_mag", 
                            "focus", "subcategory", "format", "option_descriptions", 
                            "option_content", "choices", "explanation", "answer_key", 
                            "answer_keys", "correct_answers", "correct_answer", 
                            "answer_indices", "answer_index", "answer_choice", 
                            "answer_idx", "answer_options"]:
                    if key in item:
                        info[key] = item[key]
                
                sample = {
                    "id": f"{session_name}_{item.get('qa_id', '')}",
                    "image": new_image_path,
                    "conversations": [
                        {"from": "human", "value": human_content},
                        {"from": "gpt", "value": gpt_content}
                    ],
                    "info": info
                }
                samples.append(sample)
                session_count += 1
                stats["success"] += 1
        
        print(f"Processing complete: {session_dir.name}, records: {session_count}")
    
    # Save results
    save_results(output_path, samples, filtered_records, stats, features_dir)


def save_results(output_path, samples, filtered_records, stats, features_dir):
    """Save processing results and print statistics"""
    # Save merged results as jsonl format
    output_file = output_path / "qa_merged.jsonl"
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # Save filtered records
    if filtered_records:
        filtered_file = output_path / "qa_filtered.jsonl"
        with open(filtered_file, "w") as f:
            for record in filtered_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nFiltered records saved to: {filtered_file}")
    
    # Print statistics
    print(f"\n" + "="*60)
    print(f"Data processing statistics:")
    print(f"  Total records:    {stats['total']}")
    print(f"  Successfully processed:    {stats['success']}")
    print(f"  Filtered:      {stats['filtered']}")
    if stats['total'] > 0:
        print(f"  Success rate:      {stats['success']/stats['total']*100:.2f}%")
    print(f"="*60)
    print(f"\nConversion complete: {len(samples)} samples -> {output_file}")
    print(f"Feature files copied to: {features_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python qa.py <input_dir> <output_dir>")
        sys.exit(1)
    
    batch_convert_sessions(sys.argv[1], sys.argv[2])
