import argparse
import ast
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI, APITimeoutError


def log(msg: str) -> None:
    """Print and flush immediately to ensure real-time display in non-terminal environments"""
    print(msg)
    sys.stdout.flush()


MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 60
MAX_CONCURRENT_REQUESTS = 5
API_TIMEOUT = 90


def try_parse_json_from_text(text: str) -> List[Dict[str, Any]]:
    """Try parsing plain-text list[dict] returned by the judge model"""
    if not text:
        return []
    try:
        cleaned_text = re.sub(r"```[a-zA-Z]*\n?|```", "", text).strip()
        parsed = ast.literal_eval(cleaned_text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        pass
    return []


def extract_claims_from_text(text: str) -> List[str]:
    claims: List[str] = []
    if not text:
        return claims
    for line in text.strip().splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if len(cleaned) > 5:
            claims.append(cleaned)
    return claims


def extract_scores(items: List[Dict[str, Any]]) -> List[float]:
    scores: List[float] = []
    for item in items:
        raw = item.get("score", 0)
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    return scores


def average(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


async def gpt_call_safe(
    messages: List[Dict[str, str]],
    client: OpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    stage: str = "request",
    output_json: bool = False,
    max_retries: int = MAX_RETRIES,
) -> Any:
    async with semaphore:
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                req_start = time.time()
                log(f"[API][{stage}] start (attempt {attempt + 1}/{max_retries})")
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                )
                elapsed = time.time() - req_start
                content = response.choices[0].message.content or ""
                log(f"[API][{stage}] done in {elapsed:.1f}s")
                return try_parse_json_from_text(content) if output_json else content
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    wait_time = min(INITIAL_RETRY_DELAY * (2**attempt), MAX_RETRY_DELAY)
                    log(
                        f"[WARN] API error [{stage}] ({attempt + 1}/{max_retries}): {exc}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    log(f"[ERROR] API failed after retries [{stage}]: {exc}")

        msg = str(last_exception) if last_exception else "Unknown error"
        raise RuntimeError(f"API call failed after {max_retries} retries: {msg}")


async def evaluate_wsi_metrics(
    prediction: str,
    ground_truth: str,
    client: OpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    # Step 1: claim extraction
    extract_msg = [
        {
            "role": "system",
            "content": "Extract all factual claims from the pathology dialogue as a simple list of strings.",
        },
        {"role": "user", "content": f"Dialogue: {prediction}"},
    ]
    claims_text = await gpt_call_safe(
        extract_msg, client, semaphore, model_name, stage="claim_extraction"
    )
    claims = extract_claims_from_text(claims_text)

    precision_sys = """
Please act as an impartial judge and evaluate the correctness of the AI assistant's pathology dialogue for each claim based on the following scoring criteria. Provide an explanation for each evaluation and assign a score.
Scoring Criteria:
- 1: The information in the pathology dialogue is completely correct regarding the claim.
- 0.7: The information is mostly correct and closely aligns with the claim.
- 0.3: The claim is mentioned but contains errors in the core content (e.g., mistakes in differentiation degree or malignancy).
- 0: The information in the pathology dialogue is completely incorrect regarding the claim.
Output Requirements:
Please output your evaluations as a list of dictionaries in plain text format (not JSON). The format should be as follows:
[
  {
    "claim": "Original claim1",
    "explanation": "Explanation for the score",
    "score": 1 or 0.7 or 0.3 or 0
  },
  {
    "claim": "Original claim2",
    "explanation": "Explanation for the score",
    "score": 1 or 0.7 or 0.3 or 0
  },
  ...
]
"""
    precision_user = f"Ground Truth: {ground_truth}\nClaims to evaluate: {claims}"
    precision_results = await gpt_call_safe(
        [{"role": "system", "content": precision_sys}, {"role": "user", "content": precision_user}],
        client,
        semaphore,
        model_name,
        stage="precision",
        output_json=True,
    )

    relevance_sys = """
Please act as an impartial judge and evaluate the relevance of the original ground truth answer to each claim derived from the model's answer. Provide an explanation for each evaluation and assign a score based on the following criteria.
Scoring Criteria:
- 1: The content in the ground truth answer is completely relevant to the claim.
- 0.7: The content is mostly relevant but has minor omissions or deviations.
- 0.3: The content is partially relevant with significant omissions or irrelevant information.
- 0: The content in the ground truth answer is not relevant to the claim.
Output Requirements:
Please output your evaluations as a list of dictionaries in plain text format (not JSON). The format should be as follows:
[
  {
    "claim": "Original claim1",
    "explanation": "Explanation for the score",
    "score": 1 or 0.7 or 0.3 or 0
  },
  {
    "claim": "Original claim2",
    "explanation": "Explanation for the score",
    "score": 1 or 0.7 or 0.3 or 0
  },
  ...
]
"""
    relevance_user = f"Ground Truth: {ground_truth}\nClaims from model: {claims}"
    relevance_results = await gpt_call_safe(
        [{"role": "system", "content": relevance_sys}, {"role": "user", "content": relevance_user}],
        client,
        semaphore,
        model_name,
        stage="relevance",
        output_json=True,
    )

    prec_score = average(extract_scores(precision_results))
    rel_score = average(extract_scores(relevance_results))
    return prec_score, rel_score, precision_results, relevance_results, claims


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
            except json.JSONDecodeError as exc:
                log(f"[WARN] Skip invalid JSON line {idx}: {exc}")
    return data


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-ended JSONL with WSI-Precision/WSI-Relevance.")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL path.")
    parser.add_argument("--output-jsonl", default="", help="Output JSONL path with per-sample WSI metrics.")
    parser.add_argument("--summary-json", default="", help="Output summary JSON path.")
    parser.add_argument("--api-key", required=True, help="Judge API key.")
    parser.add_argument("--api-url", required=True, help="Judge API base url.")
    parser.add_argument("--api-model", required=True, help="Judge model name.")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENT_REQUESTS, help="Max concurrent requests.")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key.")

    input_path = args.input_jsonl
    output_path = args.output_jsonl or input_path.replace(".jsonl", ".wsi_eval.jsonl")
    summary_path = args.summary_json or input_path.replace(".jsonl", ".wsi_summary.json")

    client = OpenAI(api_key=args.api_key, base_url=args.api_url, timeout=API_TIMEOUT)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    # Test API connectivity
    log(f"[INFO] API endpoint: {args.api_url}")
    log(f"[INFO] Model: {args.api_model}")
    log(f"[INFO] Testing API connectivity...")
    try:
        test_resp = client.chat.completions.create(
            model=args.api_model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        log(f"[INFO] API test OK: {test_resp.choices[0].message.content!r}")
    except Exception as exc:
        log(f"[ERROR] API test FAILED: {exc}")
        log("[ERROR] Please check --api-url, --api-key, --api-model and network connectivity.")
        return

    data = read_jsonl(input_path)
    if not data:
        raise ValueError(f"No valid samples found in: {input_path}")
    log(f"[INFO] Loaded {len(data)} samples from {input_path}")
    log(f"[INFO] Max concurrency: {args.max_concurrency}, Timeout: {API_TIMEOUT}s")
    log(f"[INFO] Starting evaluation...\n")

    completed_count = 0
    start_time = time.time()

    async def _eval_one(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
        nonlocal completed_count
        qid = item.get("question_id", "N/A")
        log(f"[start] #{idx+1}/{len(data)} question_id={qid}")
        prediction = item.get("prediction", "")
        ground_truth = item.get("ground_truth", "")
        if not prediction or not ground_truth:
            completed_count += 1
            out = dict(item)
            out["wsi_eval_error"] = "missing prediction or ground_truth"
            out["wsi_precision"] = 0.0
            out["wsi_relevance"] = 0.0
            out["wsi_claims"] = []
            out["wsi_precision_detail"] = []
            out["wsi_relevance_detail"] = []
            log(f"[{completed_count}/{len(data)}] #{idx+1} skipped (missing prediction/ground_truth)")
            return out
        try:
            p, r, pd, rd, claims = await evaluate_wsi_metrics(
                prediction=prediction,
                ground_truth=ground_truth,
                client=client,
                semaphore=semaphore,
                model_name=args.api_model,
            )
            completed_count += 1
            elapsed = time.time() - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            eta = (len(data) - completed_count) / rate if rate > 0 else 0
            out = dict(item)
            out["wsi_precision"] = round(p, 4)
            out["wsi_relevance"] = round(r, 4)
            out["wsi_claims"] = claims
            out["wsi_precision_detail"] = pd
            out["wsi_relevance_detail"] = rd
            out["wsi_eval_error"] = ""
            log(f"[{completed_count}/{len(data)}] #{idx+1} done: "
                f"WSI-P={out['wsi_precision']}, WSI-R={out['wsi_relevance']}  "
                f"({rate:.1f} samples/min, ETA {eta/60:.0f}min)")
            return out
        except Exception as exc:
            completed_count += 1
            out = dict(item)
            out["wsi_eval_error"] = str(exc)
            out["wsi_precision"] = 0.0
            out["wsi_relevance"] = 0.0
            out["wsi_claims"] = []
            out["wsi_precision_detail"] = []
            out["wsi_relevance_detail"] = []
            log(f"[{completed_count}/{len(data)}] #{idx+1} failed: {exc}")
            return out

    # Sample-level worker pool
    worker_num = max(1, min(args.max_concurrency, len(data)))
    work_queue: asyncio.Queue[Tuple[int, Dict[str, Any]]] = asyncio.Queue()
    for i, item in enumerate(data):
        work_queue.put_nowait((i, item))

    results: List[Optional[Dict[str, Any]]] = [None] * len(data)

    async def _worker(worker_id: int) -> None:
        while True:
            try:
                idx, item = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[idx] = await _eval_one(item, idx)
            finally:
                work_queue.task_done()

    workers = [asyncio.create_task(_worker(wid)) for wid in range(worker_num)]
    await asyncio.gather(*workers)
    final_results: List[Dict[str, Any]] = [x for x in results if x is not None]

    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    overall_p = average([float(x.get("wsi_precision", 0.0)) for x in final_results])
    overall_r = average([float(x.get("wsi_relevance", 0.0)) for x in final_results])

    by_focus: Dict[str, Dict[str, float]] = {}
    by_subcategory: Dict[str, Dict[str, float]] = {}

    focus_bucket: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    sub_bucket: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for x in final_results:
        info = x.get("info", {}) if isinstance(x.get("info"), dict) else {}
        focus = str(info.get("focus", "unknown"))
        subcat = str(info.get("subcategory", "unknown"))
        focus_bucket[focus].append((float(x.get("wsi_precision", 0.0)), float(x.get("wsi_relevance", 0.0))))
        sub_bucket[subcat].append((float(x.get("wsi_precision", 0.0)), float(x.get("wsi_relevance", 0.0))))

    for k, vals in focus_bucket.items():
        by_focus[k] = {
            "count": len(vals),
            "wsi_precision_mean": round(average([v[0] for v in vals]), 4),
            "wsi_relevance_mean": round(average([v[1] for v in vals]), 4),
        }
    for k, vals in sub_bucket.items():
        by_subcategory[k] = {
            "count": len(vals),
            "wsi_precision_mean": round(average([v[0] for v in vals]), 4),
            "wsi_relevance_mean": round(average([v[1] for v in vals]), 4),
        }

    summary = {
        "input_jsonl": input_path,
        "num_samples": len(final_results),
        "overall": {
            "wsi_precision_mean": round(overall_p, 4),
            "wsi_relevance_mean": round(overall_r, 4),
        },
        "by_focus": by_focus,
        "by_subcategory": by_subcategory,
        "output_jsonl": output_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    log(f"\n=== WSI Evaluation Summary === (total time: {total_time/60:.1f} min)")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"\nPer-sample output: {output_path}")
    log(f"Summary output: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
