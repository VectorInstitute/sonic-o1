#!/usr/bin/env python3
import json
import os
from pathlib import Path

def count_empty_demographics(directory):
    """Count entries with empty demographics arrays in all JSON files."""
    
    results = {}
    total_empty = 0
    total_entries = 0
    
    # Get all JSON files in the directory
    json_files = sorted(Path(directory).glob("*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        topic_name = data.get("topic_name", "Unknown")
        entries = data.get("entries", [])
        
        empty_count = 0
        for entry in entries:
            demographics = entry.get("demographics", [])
            if not demographics or demographics == []:
                empty_count += 1
        
        results[json_file.name] = {
            "topic_name": topic_name,
            "total_entries": len(entries),
            "empty_demographics": empty_count,
            "percentage": round((empty_count / len(entries) * 100), 2) if entries else 0
        }
        
        total_empty += empty_count
        total_entries += len(entries)
    
    # Print results
    print("=" * 80)
    print("EMPTY DEMOGRAPHICS REPORT")
    print("=" * 80)
    print()
    
    for filename, stats in results.items():
        print(f"{filename}")
        print(f"  Topic: {stats['topic_name']}")
        print(f"  Total Entries: {stats['total_entries']}")
        print(f"  Empty Demographics: {stats['empty_demographics']}")
        print(f"  Percentage: {stats['percentage']}%")
        print()
    
    print("=" * 80)
    print(f"TOTAL SUMMARY")
    print(f"  Total Entries Across All Topics: {total_entries}")
    print(f"  Total Empty Demographics: {total_empty}")
    print(f"  Overall Percentage: {round((total_empty / total_entries * 100), 2) if total_entries else 0}%")
    print("=" * 80)

if __name__ == "__main__":
    # Use the path from your screenshot
    directory = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa/task3_temporal_localization"
    count_empty_demographics(directory)