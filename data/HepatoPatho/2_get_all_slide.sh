#!/bin/bash

WSI_DIR="${1:-./data/wsi}"
CASE_INFO_JSON="${2:-./data/case_info.json}"
CLUSTER_GEOMETRY="${3:-./data/cluster_geometry.json}"
OUTPUT_DIR="${4:-./output}"
MAX_PARALLEL="${5:-4}"  # 默认最大并发数为4


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/2_get_per_slide.py"

# 已完成文件列表
COMPLETED_FILE="$OUTPUT_DIR/completed_wsi.txt"
touch "$COMPLETED_FILE"


# 读取已完成的文件列表（去重）
declare -A completed_files
if [ -f "$COMPLETED_FILE" ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && completed_files["$line"]=1
    done < "$COMPLETED_FILE"
    echo "Loaded ${#completed_files[@]} completed files from $COMPLETED_FILE"
fi

# 安全追加到已完成文件列表的函数
append_completed() {
    local wsi_basename="$1"
    if command -v flock &> /dev/null; then
        # 使用flock加锁
        (
            flock -n 9 || exit 1
            echo "$wsi_basename" >> "$COMPLETED_FILE"
        ) 9>"$COMPLETED_FILE.lock"
    else
        echo "$wsi_basename" >> "$COMPLETED_FILE"
    fi
}

# 查找所有WSI文件并过滤已完成的
ALL_WSI_FILES=($(find "$WSI_DIR" -type f -name "*.svs" | sort))
WSI_FILES=()

for wsi_file in "${ALL_WSI_FILES[@]}"; do
    wsi_basename=$(basename "$wsi_file" .svs)
    if [ -z "${completed_files[$wsi_basename]}" ]; then
        WSI_FILES+=("$wsi_file")
    fi
done

echo "Found ${#ALL_WSI_FILES[@]} total WSI files, ${#completed_files[@]} already completed, ${#WSI_FILES[@]} remaining to process"

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

process_wsi() {
    local wsi_file="$1"
    local wsi_basename=$(basename "$wsi_file" .svs)
    local log_file="$LOG_DIR/${wsi_basename}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]Processing: $wsi_basename" | tee -a "$log_file"
    
    python "$PYTHON_SCRIPT" \
        --wsi-path "$wsi_file" \
        --case-info-json "$CASE_INFO_JSON" \
        --cluster-geometry "$CLUSTER_GEOMETRY" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Completed: $wsi_basename" | tee -a "$log_file"
        # 追加到已完成文件列表
        append_completed "$wsi_basename"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Failed: $wsi_basename (exit code: $exit_code)" | tee -a "$log_file"
    fi
    
    return $exit_code
}

export -f process_wsi append_completed
export PYTHON_SCRIPT CASE_INFO_JSON CLUSTER_GEOMETRY OUTPUT_DIR LOG_DIR COMPLETED_FILE

if [ ${#WSI_FILES[@]} -eq 0 ]; then
    echo "✅ All WSI files have been processed!"
    exit 0
fi

if command -v parallel &> /dev/null; then
    echo "📦 Using GNU parallel for parallel processing..."
    printf '%s\n' "${WSI_FILES[@]}" | \
        parallel -j "$MAX_PARALLEL" process_wsi {}
else
    echo "📦 Using xargs for parallel processing..."
    printf '%s\n' "${WSI_FILES[@]}" | \
        xargs -n 1 -P "$MAX_PARALLEL" -I {} bash -c 'process_wsi "{}"'
fi

# 清理锁文件
rm -f "$COMPLETED_FILE.lock"

