#!/usr/bin/env python3
"""
Statistical analysis of choice_detailed.jsonl file
- Calculate accuracy by subcategory
- Calculate accuracy by format (single/multi)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_score(item):
    """
    Calculate score for single record:
    - Single choice: 1 if correct, 0 otherwise
    - Multiple choice: 1 if fully correct, 0.5 if partially correct (pred is subset of answer), 0 if wrong
    """
    format_type = item.get('info', {}).get('format', 'unknown')
    pred_letters = set(item.get('pred_letters', []))
    ref_letters = set(item.get('ref_letters', []))

    if format_type == 'multi':
        if pred_letters == ref_letters:
            return 1.0
        # Prediction is subset of correct answer, but not complete
        if pred_letters and pred_letters.issubset(ref_letters):
            return 0.5
        # Prediction contains wrong options or is empty
        return 0.0
    else:
        # Single choice
        return 1.0 if item.get('exact_match', False) else 0.0


def calculate_accuracy(data):
    """Calculate accuracy statistics (multi-choice uses weighted score)"""
    # Statistics by subcategory
    subcategory_stats = defaultdict(lambda: {'total': 0, 'score': 0.0})
    
    # Statistics by format
    format_stats = defaultdict(lambda: {'total': 0, 'score': 0.0})
    
    # Statistics by major category (1.1, 1.2 -> 1; 2.1, 2.2 -> 2)
    major_category_stats = defaultdict(lambda: {'total': 0, 'score': 0.0})
    
    # Statistics by major category + belong_level
    major_level_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'score': 0.0}))
    
    # Statistics by major category + format
    major_format_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'score': 0.0}))
    
    # Statistics by belong_level
    belong_level_stats = defaultdict(lambda: {'total': 0, 'score': 0.0})
    
    # Overall statistics
    total_count = 0
    total_score = 0.0
    
    for item in data:
        subcategory = item.get('info', {}).get('subcategory', 'unknown')
        format_type = item.get('info', {}).get('format', 'unknown')
        belong_level = item.get('info', {}).get('belong_level', 'unknown')
        
        # Extract major category (part before decimal point)
        major_cat = subcategory.split('.')[0] if '.' in subcategory else subcategory
        
        # Calculate weighted score
        score = compute_score(item)
        
        total_count += 1
        total_score += score
        
        # By subcategory
        subcategory_stats[subcategory]['total'] += 1
        subcategory_stats[subcategory]['score'] += score
        
        # By format
        format_stats[format_type]['total'] += 1
        format_stats[format_type]['score'] += score
        
        # By major category
        major_category_stats[major_cat]['total'] += 1
        major_category_stats[major_cat]['score'] += score
        
        # By major category + belong_level
        major_level_stats[major_cat][belong_level]['total'] += 1
        major_level_stats[major_cat][belong_level]['score'] += score
        
        # By major category + format
        major_format_stats[major_cat][format_type]['total'] += 1
        major_format_stats[major_cat][format_type]['score'] += score
        
        # By belong_level
        belong_level_stats[belong_level]['total'] += 1
        belong_level_stats[belong_level]['score'] += score
    
    return {
        'total': {'count': total_count, 'score': total_score},
        'subcategory': subcategory_stats,
        'format': format_stats,
        'major_category': major_category_stats,
        'major_level': major_level_stats,
        'major_format': major_format_stats,
        'belong_level': belong_level_stats
    }


def print_statistics(stats):
    """Print statistics results"""
    print("=" * 80)
    print("Evaluation Statistics")
    print("=" * 80)
    print("(Multi-choice scoring: fully correct=1, partially correct=0.5, wrong=0)")
    
    # Overall statistics
    total = stats['total']
    total_acc = (total['score'] / total['count'] * 100) if total['count'] > 0 else 0
    print(f"\n【Overall】")
    print(f"  Total samples: {total['count']}")
    print(f"  Total score: {total['score']:.1f}")
    print(f"  Accuracy: {total_acc:.2f}%")
    
    # By format
    print(f"\n【By Format】")
    format_stats = stats['format']
    for format_type in sorted(format_stats.keys()):
        fmt_stat = format_stats[format_type]
        fmt_acc = (fmt_stat['score'] / fmt_stat['total'] * 100) if fmt_stat['total'] > 0 else 0
        format_name = "Single" if format_type == "single" else "Multi" if format_type == "multi" else format_type
        print(f"  {format_name}:")
        print(f"    Samples: {fmt_stat['total']}")
        print(f"    Score: {fmt_stat['score']:.1f}")
        print(f"    Accuracy: {fmt_acc:.2f}%")
    
    # By subcategory
    print(f"\n【By Subcategory】")
    subcategory_stats = stats['subcategory']
    for subcat in sorted(subcategory_stats.keys()):
        subcat_stat = subcategory_stats[subcat]
        subcat_acc = (subcat_stat['score'] / subcat_stat['total'] * 100) if subcat_stat['total'] > 0 else 0
        print(f"  {subcat}:")
        print(f"    Samples: {subcat_stat['total']}")
        print(f"    Score: {subcat_stat['score']:.1f}")
        print(f"    Accuracy: {subcat_acc:.2f}%")
    
    # By major category (with belong_level and format breakdown)
    print(f"\n【By Major Category】")
    major_category_stats = stats['major_category']
    major_level_stats = stats['major_level']
    major_format_stats = stats['major_format']
    for major_cat in sorted(major_category_stats.keys()):
        cat_stat = major_category_stats[major_cat]
        cat_acc = (cat_stat['score'] / cat_stat['total'] * 100) if cat_stat['total'] > 0 else 0
        print(f"  Category {major_cat}:")
        print(f"    Samples: {cat_stat['total']}")
        print(f"    Score: {cat_stat['score']:.1f}")
        print(f"    Accuracy: {cat_acc:.2f}%")
        
        # By belong_level
        level_stats = major_level_stats[major_cat]
        if level_stats:
            print(f"    ---- By belong_level ----")
            for level in sorted(level_stats.keys()):
                lv_stat = level_stats[level]
                lv_acc = (lv_stat['score'] / lv_stat['total'] * 100) if lv_stat['total'] > 0 else 0
                print(f"    {level}:")
                print(f"      Samples: {lv_stat['total']}")
                print(f"      Score: {lv_stat['score']:.1f}")
                print(f"      Accuracy: {lv_acc:.2f}%")
        
        # By format
        fmt_stats = major_format_stats[major_cat]
        if fmt_stats:
            print(f"    ---- By Format ----")
            for format_type in sorted(fmt_stats.keys()):
                fm_stat = fmt_stats[format_type]
                fm_acc = (fm_stat['score'] / fm_stat['total'] * 100) if fm_stat['total'] > 0 else 0
                format_name = "Single" if format_type == "single" else "Multi" if format_type == "multi" else format_type
                print(f"    {format_name}:")
                print(f"      Samples: {fm_stat['total']}")
                print(f"      Score: {fm_stat['score']:.1f}")
                print(f"      Accuracy: {fm_acc:.2f}%")
    
    # By belong_level
    print(f"\n【By belong_level】")
    belong_level_stats = stats['belong_level']
    for level in sorted(belong_level_stats.keys()):
        level_stat = belong_level_stats[level]
        level_acc = (level_stat['score'] / level_stat['total'] * 100) if level_stat['total'] > 0 else 0
        print(f"  {level}:")
        print(f"    Samples: {level_stat['total']}")
        print(f"    Score: {level_stat['score']:.1f}")
        print(f"    Accuracy: {level_acc:.2f}%")
    
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python stat_choice.py <choice_detailed.jsonl>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Loading file: {file_path}")
    data = load_jsonl(file_path)
    print(f"Loaded {len(data)} records")
    
    print("Calculating statistics...")
    stats = calculate_accuracy(data)
    
    print_statistics(stats)


if __name__ == '__main__':
    main()
