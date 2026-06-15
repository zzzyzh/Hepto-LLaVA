#!/bin/bash

# ============================================================
# I. 参数配置 (CONFIG)
# ============================================================

WSI_DIR="${1:-./data/wsi}"
CASE_INFO_JSON="${2:-./data/case_info.json}"
CLUSTER_GEOMETRY="${3:-./data/cluster_geometry.json}"
OUTPUT_DIR="${4:-./output}"
MAX_PARALLEL="${5:-4}"
TARGET_COMPLETED="${6:-75}"   # ← 新含义：目标成功数量（不是最大处理数）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/6_hepato_bench.py"

COMPLETED_FILE="$OUTPUT_DIR/completed_wsi.txt"
LOG_DIR="$OUTPUT_DIR/pipeline_logs"
FAILURE_RECORD="$LOG_DIR/failed_tasks.list"
SKIPPED_LIST="$LOG_DIR/skipped.list"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
touch "$COMPLETED_FILE"
touch "$FAILURE_RECORD"
touch "$SKIPPED_LIST"

# ============================================================
# II. 辅助函数 (保持不变)
# ============================================================

record_status() {
    local wsi_basename="$1"
    local status="$2"
    (
        flock -x 9 || exit 1
        if [ "$status" == "success" ]; then
            echo "$wsi_basename" >> "$COMPLETED_FILE"
        else
            echo "$wsi_basename" >> "$FAILURE_RECORD"
        fi
    ) 9>"$OUTPUT_DIR/status.lock"
}

record_skip() {
    local wsi_basename="$1"
    (
        flock -x 9 || exit 1
        echo "$wsi_basename" >> "$SKIPPED_LIST"
    ) 9>"$OUTPUT_DIR/skipped.lock"
}

get_fail_count() {
    wc -l < "$FAILURE_RECORD"
}

process_wsi() {
    local wsi_file="$1"
    local wsi_basename=$(basename "$wsi_file" .svs)
    local log_file="$LOG_DIR/${wsi_basename}.log"
    
    local current_fails=$(get_fail_count)
    if [ "$current_fails" -ge 10 ]; then
        echo "[$(date '+%H:%M:%S')] 🛑 Circuit Breaker: Skipping $wsi_basename"
        return 1
    fi

    echo "[$(date '+%H:%M:%S')] 🚀 Processing: $wsi_basename" | tee -a "$log_file"
    
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
    elif [ $exit_code -eq 42 ]; then
        echo "[$(date '+%H:%M:%S')] ⏭️ Skipped (no cluster): $wsi_basename" | tee -a "$log_file"
        record_skip "$wsi_basename"
    else
        echo "[$(date '+%H:%M:%S')] ❌ Failed: $wsi_basename (Code: $exit_code)" | tee -a "$log_file"
        record_status "$wsi_basename" "fail"
    fi
    
    return $exit_code
}

export -f process_wsi record_status get_fail_count record_skip
export PYTHON_SCRIPT CASE_INFO_JSON CLUSTER_GEOMETRY OUTPUT_DIR LOG_DIR \
       COMPLETED_FILE FAILURE_RECORD SKIPPED_LIST

# ============================================================
# III. 动态筛选：直到 completed 达到 TARGET
# ============================================================

# 读取当前已完成、跳过列表
declare -A skipped_map completed_map

while IFS= read -r line; do
    [ -z "$line" ] && continue
    skipped_map["$line"]=1
done < "$SKIPPED_LIST"

while IFS= read -r line; do
    [ -z "$line" ] && continue
    completed_map["$line"]=1
done < "$COMPLETED_FILE"

CURRENT_COMPLETED=${#completed_map[@]}
NEED=$(( TARGET_COMPLETED - CURRENT_COMPLETED ))

if [ $NEED -le 0 ]; then
    echo "🎉 Already have $CURRENT_COMPLETED completed (target: $TARGET_COMPLETED). Exiting."
    exit 0
fi

echo "🎯 Target: $TARGET_COMPLETED completed | Current: $CURRENT_COMPLETED | Need: $NEED"

# 获取所有 .svs 文件，并过滤掉已跳过的（completed 的可以重试失败项）
ALL_WSI=($(find "$WSI_DIR" -type f -name "*.svs" | sort))
CANDIDATE_WSI=()

for f in "${ALL_WSI[@]}"; do
    base=$(basename "$f" .svs)
    # 跳过的不再处理；已完成的也不再处理（避免重复）
    if [[ -z "${skipped_map[$base]}" ]] && [[ -z "${completed_map[$base]}" ]]; then
        CANDIDATE_WSI+=("$f")
    fi
done

if [ ${#CANDIDATE_WSI[@]} -eq 0 ]; then
    echo "⚠️ No candidate WSI left to process (all skipped or completed)."
    exit 0
fi

# 取前 NEED 个候选
if [ ${#CANDIDATE_WSI[@]} -gt $NEED ]; then
    TODO_WSI=("${CANDIDATE_WSI[@]:0:$NEED}")
else
    TODO_WSI=("${CANDIDATE_WSI[@]}")
fi

echo "📋 Will process ${#TODO_WSI[@]} WSI files (to reach target of $TARGET_COMPLETED completed)."

# ============================================================
# IV. 并行执行
# ============================================================

if command -v parallel &> /dev/null; then
    printf '%s\n' "${TODO_WSI[@]}" | parallel -j "$MAX_PARALLEL" --halt soon,fail=20% process_wsi {}
else
    printf '%s\n' "${TODO_WSI[@]}" | xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'process_wsi "{}"'
fi

# ============================================================
# V. 最终统计
# ============================================================

FINAL_COMPLETED=$(wc -l < "$COMPLETED_FILE")
FINAL_FAILS=$(get_fail_count)
SKIPPED_COUNT=$(wc -l < "$SKIPPED_LIST")

rm -f "$OUTPUT_DIR/status.lock" "$OUTPUT_DIR/skipped.lock"

echo ""
echo "📊 Final Stats:"
echo "   ✅ Completed: $FINAL_COMPLETED"
echo "   ❌ Failed:    $FINAL_FAILS"
echo "   ⏭️ Skipped:   $SKIPPED_COUNT"

if [ "$FINAL_COMPLETED" -ge "$TARGET_COMPLETED" ]; then
    echo "🎯 Target of $TARGET_COMPLETED completed reached!"
    exit 0
else
    echo "ℹ️ Target not reached (only $FINAL_COMPLETED/$TARGET_COMPLETED), but no more candidates."
    exit 0
fi