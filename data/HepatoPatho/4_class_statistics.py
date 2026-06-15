import json
import os
import glob
from collections import defaultdict

def count_qa_categories():
    # 1. 定义映射关系（用于统计时的归类标准）
    focus_to_category = {
        "architecture": ("Morphological Analysis", "Regional Structure Description"),
        "stroma": ("Morphological Analysis", "Regional Structure Description"),
        "inflammation": ("Morphological Analysis", "Specific Feature Description"),
        "cytology": ("Morphological Analysis", "Specific Feature Description"),
        "necrosis": ("Morphological Analysis", "Specific Feature Description"),
        "nuclear_detail": ("Morphological Analysis", "Specific Feature Description"),
        "macro_architecture": ("Morphological Analysis", "Global Morphology Description"),
        "global_context": ("Morphological Analysis", "Global Morphology Description"),
        "diagnosis": ("Diagnosis", "Histological Typing") # 示例映射，你可以根据需要扩充
    }

    # 初始化统计字典
    # 结构: stats[category][subclass] = count
    stats = defaultdict(lambda: defaultdict(int))
    total_files = 0
    total_qa_count = 0

    # 2. 设定路径
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    pattern = os.path.join(root_dir, "session_*", "final_integrated_results.json")
    file_list = glob.glob(pattern)

    if not file_list:
        print("未找到任何文件，请检查路径。")
        return

    print(f"正在分析 {len(file_list)} 个文件...")

    # 3. 遍历并统计
    for file_path in file_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            total_files += 1

            # 收集该文件中所有的 QA 字典
            all_qa_dicts = []
            if "wsi_results" in data and "caption_qa" in data["wsi_results"]:
                all_qa_dicts.append(data["wsi_results"]["caption_qa"])
            
            if "clusters" in data:
                for cluster in data["clusters"]:
                    if "cluster_caption_qa" in cluster:
                        all_qa_dicts.append(cluster["cluster_caption_qa"])
                    if "rois" in cluster:
                        for roi in cluster["rois"]:
                            if "caption_qa" in roi:
                                all_qa_dicts.append(roi["caption_qa"])

            # 统计每个 QA 条目
            for qa_dict in all_qa_dicts:
                for q_item in qa_dict.values():
                    focus = q_item.get("focus")
                    if focus in focus_to_category:
                        cat, subcat = focus_to_category[focus]
                        stats[cat][subcat] += 1
                        total_qa_count += 1

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 4. 打印统计报告
    print("\n" + "="*45)
    print(f"统计报告")
    print(f"处理文件总数: {total_files}")
    print(f"识别到的 QA 总数: {total_qa_count}")
    print("="*45)

    for cat, subcategories in stats.items():
        cat_total = sum(subcategories.values())
        print(f"\n【大类: {cat}】 (总计: {cat_total})")
        print("-" * 30)
        for subcat, count in subcategories.items():
            percentage = (count / cat_total * 100) if cat_total > 0 else 0
            print(f"  - {subcat.ljust(30)}: {count} 条 ({percentage:.1f}%)")
    
    print("="*45)

if __name__ == "__main__":
    count_qa_categories()