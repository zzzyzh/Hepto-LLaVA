#!/bin/bash

# ============================================================
# I. 参数配置 (CONFIG)
# ============================================================

SESSIONS_ROOT="${1:-./data/sessions}"
CASE_INFO_JSON="${2:-./data/case_info.json}"
OUTPUT_ROOT="${3:-./output}"
MAX_PARALLEL="${4:-6}"

# --- 熔断配置（仅基于累计失败数）---
MAX_TOTAL_FAILURES=10  # 可恢复错误（如 API 失败）的累计上限

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/7_caption.py"

COMPLETED_FILE="$OUTPUT_ROOT/completed_caption_sessions.txt"
LOG_DIR="$OUTPUT_ROOT/caption_logs"
FAILURE_RECORD="$LOG_DIR/failed_caption_sessions.list"  # 仅记录可恢复失败

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$LOG_DIR"
touch "$COMPLETED_FILE"
touch "$FAILURE_RECORD"

# ============================================================
# II. 辅助函数 (UTILITIES)
# ============================================================

record_status() {
    local session_name="$1"
    local status="$2" # "success" or "fail"
    
    (
        flock -x 9 || exit 1
        if [ "$status" == "success" ]; then
            echo "$session_name" >> "$COMPLETED_FILE"
        else
            echo "$session_name" >> "$FAILURE_RECORD"
        fi
    ) 9>"$OUTPUT_ROOT/status.lock"
}

get_fail_count() {
    wc -l < "$FAILURE_RECORD" 2>/dev/null || echo 0
}

process_session() {
    local session_dir="$1"
    local session_name=$(basename "$session_dir")
    local log_file="$LOG_DIR/${session_name}.log"
    
    # 检查是否因可恢复错误失败太多（熔断）
    local current_fails
    current_fails=$(get_fail_count)
    if [ "$current_fails" -ge "$MAX_TOTAL_FAILURES" ]; then
        echo "[$(date '+%H:%M:%S')] 🛑 Circuit breaker: skipping $session_name (too many recoverable failures)" | tee -a "$log_file"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] 🚀 Processing caption for session: $session_name" | tee -a "$log_file"
    
    python "$PYTHON_SCRIPT" \
        --session_dir "$session_dir" \
        --case_json "$CASE_INFO_JSON" \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        # 验证输出文件存在
        if [ -f "$session_dir/captions.jsonl" ]; then
            echo "[$(date '+%H:%M:%S')] ✅ Success: $session_name" | tee -a "$log_file"
            record_status "$session_name" "success"
        else
            echo "[$(date '+%H:%M:%S')] ⚠️ Warning: Exit code 0 but captions.jsonl missing!" | tee -a "$log_file"
            record_status "$session_name" "fail"
            return 1
        fi
    else
        echo "[$(date '+%H:%M:%S')] ❌ Failed: $session_name (Code: $exit_code)" | tee -a "$log_file"
        
        # === 关键改进：区分错误类型 ===
        if grep -q "WSI ID.*not found in case info" "$log_file"; then
            echo "[$(date '+%H:%M:%S')] 💡 Permanent failure (missing annotation). Marking as completed to avoid retry." | tee -a "$log_file"
            # 标记为“成功完成”，防止下次再跑
            record_status "$session_name" "success"
        else
            # 其他错误（API、网络、超时等）视为可恢复失败，计入熔断
            record_status "$session_name" "fail"
        fi
    fi
    
    return $exit_code
}

export -f process_session record_status get_fail_count
export PYTHON_SCRIPT CASE_INFO_JSON OUTPUT_ROOT LOG_DIR COMPLETED_FILE FAILURE_RECORD MAX_TOTAL_FAILURES

# ============================================================
# III. 文件筛选 (FILTERING)
# ============================================================

declare -A completed_map
while IFS= read -r line; do
    [ -z "$line" ] && continue
    completed_map["$line"]=1
done < "$COMPLETED_FILE"

ALL_SESSIONS=()
while IFS= read -rd '' dir; do
    if [ -f "$dir/inference_results.json" ] && [ -d "$dir/features" ]; then
        ALL_SESSIONS+=("$dir")
    fi
done < <(find "$SESSIONS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

TODO_SESSIONS=()
for dir in "${ALL_SESSIONS[@]}"; do
    name=$(basename "$dir")
    if [[ -z "${completed_map[$name]}" ]]; then
        TODO_SESSIONS+=("$dir")
    fi
done

echo "📊 Caption Pipeline Status: Total=${#ALL_SESSIONS[@]}, Completed=${#completed_map[@]}, Remaining=${#TODO_SESSIONS[@]}"

if [ ${#TODO_SESSIONS[@]} -eq 0 ]; then
    echo "🎉 All caption sessions processed. Exiting."
    exit 0
fi

# ============================================================
# IV. 并行执行 (EXECUTION) —— 已移除 --halt
# ============================================================

if command -v parallel &> /dev/null; then
    echo "📦 Using GNU Parallel (Max Parallel: $MAX_PARALLEL)"
    # 注意：已移除 --halt soon,fail="20%"
    printf '%s\n' "${TODO_SESSIONS[@]}" | parallel -j "$MAX_PARALLEL" process_session {}
else
    echo "📦 Using xargs (Max Parallel: $MAX_PARALLEL)"
    printf '%s\n' "${TODO_SESSIONS[@]}" | xargs -I {} -P "$MAX_PARALLEL" bash -c 'process_session "{}"'
fi

# Final check: only count *recoverable* failures
FINAL_FAILS=$(get_fail_count)
rm -f "$OUTPUT_ROOT/status.lock"

if [ "$FINAL_FAILS" -ge "$MAX_TOTAL_FAILURES" ]; then
    echo "🚨 CAPTION PIPELINE HALTED: Too many recoverable failures ($FINAL_FAILS >= $MAX_TOTAL_FAILURES)."
    exit 1
else
    echo "🏁 Caption pipeline finished."
    exit 0
fi