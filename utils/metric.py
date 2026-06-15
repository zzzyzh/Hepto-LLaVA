"""
Evaluation metrics for model outputs.

Includes:
- Multiple choice matching (exact_match_choice)
- ROUGE-L for open-ended questions
- METEOR for open-ended questions
"""

import re
import string
from typing import List, Dict, Union, Optional
from collections import Counter


def extract_choice_letters(text: str, preserve_case: bool = True) -> List[str]:
    """Extract option letters from text."""
    if preserve_case:
        # 添加 ^[A-Z]$ 分支来匹配单个字母的情况
        pattern = r'(?<=[^a-zA-Z])[A-Z](?=[^a-zA-Z])|^[A-Z](?=[^a-zA-Z])|(?<=[^a-zA-Z])[A-Z]$|^[A-Z]$'
        letters = re.findall(pattern, text)
    else:
        # 添加 ^[A-Za-z]$ 分支来匹配单个字母的情况
        pattern = r'(?<=[^a-zA-Z])[A-Za-z](?=[^a-zA-Z])|^[A-Za-z](?=[^a-zA-Z])|(?<=[^a-zA-Z])[A-Za-z]$|^[A-Za-z]$'
        letters = [c.upper() for c in re.findall(pattern, text)]
    
    return letters


def exact_match_choice(
    prediction: str,
    reference: Union[str, List[str]],
    preserve_case: bool = True,
    require_order: bool = False
) -> Dict[str, Union[bool, float]]:
    """
    Multiple choice exact match. All reference letters must appear in prediction with matching case.
    
    Returns dict with: exact_match, score, pred_letters, ref_letters, all_found, no_extra
    """
    if isinstance(reference, str):
        ref_letters = extract_choice_letters(reference, preserve_case)
        if not ref_letters:
            ref_letters = [c for c in reference if c.isupper()]
    else:
        ref_letters = reference
    
    pred_letters = extract_choice_letters(prediction, preserve_case)
    
    if require_order:
        all_found = pred_letters[:len(ref_letters)] == ref_letters
    else:
        ref_counter = Counter(ref_letters)
        pred_counter = Counter(pred_letters)
        all_found = all(pred_counter[letter] >= count 
                       for letter, count in ref_counter.items())
    
    no_extra = set(pred_letters) == set(ref_letters)
    exact_match = all_found and no_extra
    
    return {
        "exact_match": exact_match,
        "score": 1.0 if exact_match else 0.0,
        "pred_letters": pred_letters,
        "ref_letters": ref_letters,
        "all_found": all_found,
        "no_extra": no_extra,
    }


def batch_exact_match_choice(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    preserve_case: bool = True,
    require_order: bool = False
) -> Dict[str, float]:
    """Batch compute choice exact match. Returns accuracy, num_correct, num_total."""
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} vs {len(references)}"
    
    num_correct = 0
    for pred, ref in zip(predictions, references):
        result = exact_match_choice(pred, ref, preserve_case, require_order)
        if result["exact_match"]:
            num_correct += 1
    
    return {
        "accuracy": num_correct / len(predictions) if predictions else 0.0,
        "num_correct": num_correct,
        "num_total": len(predictions),
    }


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute longest common subsequence (LCS) length using dynamic programming."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def rouge_l(
    prediction: str,
    reference: str,
    tokenize: bool = True,
    beta: float = 1.0
) -> Dict[str, float]:
    """
    ROUGE-L score based on longest common subsequence.
    Returns rouge_l_precision, rouge_l_recall, rouge_l_fmeasure.
    """
    def preprocess(text: str) -> List[str]:
        text = text.lower()
        if tokenize:
            tokens = text.split()
            tokens = [token.strip(string.punctuation) for token in tokens]
            tokens = [token for token in tokens if token]
        else:
            tokens = list(text)
        return tokens
    
    pred_tokens = preprocess(prediction)
    ref_tokens = preprocess(reference)
    
    if not pred_tokens or not ref_tokens:
        return {
            "rouge_l_precision": 0.0,
            "rouge_l_recall": 0.0,
            "rouge_l_fmeasure": 0.0,
        }
    
    lcs_len = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall > 0:
        fmeasure = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    else:
        fmeasure = 0.0
    
    return {
        "rouge_l_precision": precision,
        "rouge_l_recall": recall,
        "rouge_l_fmeasure": fmeasure,
    }


def batch_rouge_l(
    predictions: List[str],
    references: List[str],
    tokenize: bool = True,
    beta: float = 1.0
) -> Dict[str, float]:
    """Batch compute ROUGE-L scores."""
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} vs {len(references)}"
    
    total_precision = 0.0
    total_recall = 0.0
    total_fmeasure = 0.0
    
    for pred, ref in zip(predictions, references):
        scores = rouge_l(pred, ref, tokenize, beta)
        total_precision += scores["rouge_l_precision"]
        total_recall += scores["rouge_l_recall"]
        total_fmeasure += scores["rouge_l_fmeasure"]
    
    n = len(predictions)
    return {
        "rouge_l_precision": total_precision / n if n > 0 else 0.0,
        "rouge_l_recall": total_recall / n if n > 0 else 0.0,
        "rouge_l_fmeasure": total_fmeasure / n if n > 0 else 0.0,
    }


def _create_unigram_mapping(tokens: List[str]) -> Dict[str, List[int]]:
    """Create mapping from unigram to position indices."""
    mapping = {}
    for i, token in enumerate(tokens):
        if token not in mapping:
            mapping[token] = []
        mapping[token].append(i)
    return mapping


def _find_alignments(pred_tokens: List[str], ref_tokens: List[str]) -> List[tuple]:
    """Find token alignments between prediction and reference."""
    pred_mapping = _create_unigram_mapping(pred_tokens)
    ref_mapping = _create_unigram_mapping(ref_tokens)
    
    alignments = []
    pred_matched = set()
    ref_matched = set()
    
    for token in pred_mapping:
        if token in ref_mapping:
            pred_indices = pred_mapping[token]
            ref_indices = ref_mapping[token]
            
            for p_idx in pred_indices:
                if p_idx in pred_matched:
                    continue
                for r_idx in ref_indices:
                    if r_idx in ref_matched:
                        continue
                    alignments.append((p_idx, r_idx))
                    pred_matched.add(p_idx)
                    ref_matched.add(r_idx)
                    break
    
    return alignments


def _count_chunks(alignments: List[tuple]) -> int:
    """Count number of chunks in alignments. A chunk is a contiguous alignment sequence."""
    if not alignments:
        return 0
    
    alignments = sorted(alignments)
    chunks = 1
    prev_pred, prev_ref = alignments[0]
    
    for pred_idx, ref_idx in alignments[1:]:
        if pred_idx != prev_pred + 1 or ref_idx != prev_ref + 1:
            chunks += 1
        prev_pred, prev_ref = pred_idx, ref_idx
    
    return chunks


def meteor(
    prediction: str,
    reference: str,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5
) -> Dict[str, float]:
    """
    METEOR score considering precision, recall and alignment fragmentation.
    Returns meteor_score, precision, recall, fmean, penalty.
    """
    def preprocess(text: str) -> List[str]:
        text = text.lower()
        tokens = []
        current = []
        for char in text:
            if char.isalnum() or char in "'-":
                current.append(char)
            else:
                if current:
                    tokens.append(''.join(current))
                    current = []
        if current:
            tokens.append(''.join(current))
        return tokens
    
    pred_tokens = preprocess(prediction)
    ref_tokens = preprocess(reference)
    
    if not pred_tokens or not ref_tokens:
        return {
            "meteor_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fmean": 0.0,
            "penalty": 1.0,
        }
    
    alignments = _find_alignments(pred_tokens, ref_tokens)
    num_matches = len(alignments)
    
    precision = num_matches / len(pred_tokens) if pred_tokens else 0.0
    recall = num_matches / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall > 0:
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    else:
        fmean = 0.0
    
    if num_matches > 0:
        chunks = _count_chunks(alignments)
        fragmentation = chunks / num_matches
        penalty = gamma * (fragmentation ** beta)
    else:
        penalty = 0.0
    
    meteor_score = fmean * (1 - penalty)
    
    return {
        "meteor_score": meteor_score,
        "precision": precision,
        "recall": recall,
        "fmean": fmean,
        "penalty": penalty,
    }


def batch_meteor(
    predictions: List[str],
    references: List[str],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5
) -> Dict[str, float]:
    """Batch compute METEOR scores."""
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} vs {len(references)}"
    
    total_score = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_fmean = 0.0
    total_penalty = 0.0
    
    for pred, ref in zip(predictions, references):
        scores = meteor(pred, ref, alpha, beta, gamma)
        total_score += scores["meteor_score"]
        total_precision += scores["precision"]
        total_recall += scores["recall"]
        total_fmean += scores["fmean"]
        total_penalty += scores["penalty"]
    
    n = len(predictions)
    return {
        "meteor_score": total_score / n if n > 0 else 0.0,
        "precision": total_precision / n if n > 0 else 0.0,
        "recall": total_recall / n if n > 0 else 0.0,
        "fmean": total_fmean / n if n > 0 else 0.0,
        "penalty": total_penalty / n if n > 0 else 0.0,
    }


def compute_metrics(
    predictions: List[str],
    references: List[str],
    task_type: str = "open_ended",
    **kwargs
) -> Dict[str, float]:
    """
    Unified metric computation interface.
    
    task_type: "choice" for multiple choice, "open_ended" for open-ended questions.
    """
    if task_type == "choice":
        choice_metrics = batch_exact_match_choice(predictions, references, **kwargs)
        return choice_metrics
    
    elif task_type == "open_ended":
        rouge_metrics = batch_rouge_l(predictions, references)
        meteor_metrics = batch_meteor(predictions, references)
        return {
            **rouge_metrics,
            **meteor_metrics,
        }
    
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

