#!/bin/bash

# ============================================================
# I. 参数配置 (CONFIG)
# ============================================================

WSI_DIR="${1:-./data/wsi}"
CASE_INFO_JSON="${2:-./data/case_info.json}"
CLUSTER_GEOMETRY="${3:-./data/cluster_geometry.json}"
OUTPUT_DIR="${4:-./output}"
MAX_PARALLEL="${5:-32}"

# --- 新增：熔断配置 ---
MAX_TOTAL_FAILURES=10       # 允许的最大累计失败数，超过则停止整个流水线
FAILURE_RATE_LIMIT="20%"    # GNU Parallel 模式下：如果失败率超过 20%，停止任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/6_hepato_bench.py"

COMPLETED_FILE="$OUTPUT_DIR/completed_wsi.txt"
LOG_DIR="$OUTPUT_DIR/pipeline_logs"
FAILURE_RECORD="$LOG_DIR/failed_tasks.list" # 记录失败任务的文件

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
touch "$COMPLETED_FILE"
touch "$FAILURE_RECORD"

# ============================================================
# II. 辅助函数 (UTILITIES)
# ============================================================

# 加锁记录结果，防止并发冲突
record_status() {
    local wsi_basename="$1"
    local status="$2" # "success" or "fail"
    
    (
        flock -x 9 || exit 1
        if [ "$status" == "success" ]; then
            echo "$wsi_basename" >> "$COMPLETED_FILE"
        else
            echo "$wsi_basename" >> "$FAILURE_RECORD"
        fi
    ) 9>"$OUTPUT_DIR/status.lock"
}

# 检查当前失败总数
get_fail_count() {
    wc -l < "$FAILURE_RECORD"
}

# 核心处理函数
process_wsi() {
    local wsi_file="$1"
    local wsi_basename=$(basename "$wsi_file" .svs)
    local log_file="$LOG_DIR/${wsi_basename}.log"
    
    # 检查熔断：如果失败太多，直接拒绝执行新任务
    local current_fails=$(get_fail_count)
    if [ "$current_fails" -ge "$MAX_TOTAL_FAILURES" ]; then
        echo "[$(date '+%H:%M:%S')] 🛑 Circuit Breaker Active: Skipping $wsi_basename due to too many prior failures."
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] 🚀 Processing: $wsi_basename" | tee -a "$log_file"
    
    # 运行 Python 脚本
    python "$PYTHON_SCRIPT" \
        --wsi-path "$wsi_file" \
        --case-info-json "$CASE_INFO_JSON" \
        --cluster-geometry "$CLUSTER_GEOMETRY" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] ✅ Success: $wsi_basename" | tee -a "$log_file"
        record_status "$wsi_basename" "success"
    else
        echo "[$(date '+%H:%M:%S')] ❌ Failed: $wsi_basename (Code: $exit_code)" | tee -a "$log_file"
        record_status "$wsi_basename" "fail"
    fi
    
    return $exit_code
}

export -f process_wsi record_status get_fail_count
export PYTHON_SCRIPT CASE_INFO_JSON CLUSTER_GEOMETRY OUTPUT_DIR LOG_DIR COMPLETED_FILE FAILURE_RECORD MAX_TOTAL_FAILURES

# ============================================================
# III. 文件筛选 (FILTERING)
# ============================================================

declare -A completed_map
while IFS= read -r line; do
    [ -z "$line" ] && continue
    completed_map["$line"]=1
done < "$COMPLETED_FILE"

ALL_WSI=($(find "$WSI_DIR" -type f -name "*.svs" | sort))
TODO_WSI=()

for f in "${ALL_WSI[@]}"; do
    base=$(basename "$f" .svs)
    if [[ -z "${completed_map[$base]}" ]]; then
        TODO_WSI+=("$f")
    fi
done

echo "📊 Pipeline Status: Total=${#ALL_WSI[@]}, Completed=${#completed_map[@]}, Remaining=${#TODO_WSI[@]}"

if [ ${#TODO_WSI[@]} -eq 0 ]; then
    echo "🎉 All files processed. Exiting."
    exit 0
fi

# ============================================================
# IV. 并行执行 (EXECUTION)
# ============================================================



if command -v parallel &> /dev/null; then
    echo "📦 Using GNU Parallel (Max Parallel: $MAX_PARALLEL)"
    # --halt soon,fail=X% 表示如果失败率过高，尽快停止未开始的任务
    printf '%s\n' "${TODO_WSI[@]}" | parallel -j "$MAX_PARALLEL" --halt soon,fail="$FAILURE_RATE_LIMIT" process_wsi {}
else
    echo "📦 Using xargs (Max Parallel: $MAX_PARALLEL)"
    # xargs 模式下使用脚本内检查的 MAX_TOTAL_FAILURES
    printf '%s\n' "${TODO_WSI[@]}" | xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'process_wsi "{}"'
fi

# 检查最终状态
FINAL_FAILS=$(get_fail_count)
rm -f "$OUTPUT_DIR/status.lock"

if [ "$FINAL_FAILS" -ge "$MAX_TOTAL_FAILURES" ]; then
    echo "🚨 PIPELINE HALTED: Reached failure threshold ($FINAL_FAILS >= $MAX_TOTAL_FAILURES)."
    exit 1
else
    echo "🏁 Pipeline complete."
    exit 0
fi