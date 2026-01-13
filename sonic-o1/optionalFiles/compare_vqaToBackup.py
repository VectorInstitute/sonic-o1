import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

def load_all_entries(base_path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Load all entries from VQA directory.
    
    Returns:
        Dict structure: {task_name: {filename: [entries]}}
    """
    all_data = {
        'task1_summarization': {},
        'task2_mcq': {},
        'task3_temporal_localization': {}
    }
    
    for task_name in all_data.keys():
        task_path = Path(base_path) / task_name
        if not task_path.exists():
            continue
            
        for json_file in task_path.glob('*.json'):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data[task_name][json_file.name] = data.get('entries', [])
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return all_data

def compare_entries(backup_entry: Dict, current_entry: Dict, task_key: str) -> Tuple[bool, str]:
    """
    Compare two entries and determine if they're different and why.
    
    Returns:
        (is_different, reason)
    """
    # For summarization
    if task_key == 'summary':
        backup_conf = backup_entry.get('confidence', 1.0)
        current_conf = current_entry.get('confidence', 1.0)
        
        if backup_conf != current_conf:
            return True, f"confidence changed: {backup_conf} → {current_conf}"
        
        # Check if summary content changed
        if backup_entry.get('summary_short') != current_entry.get('summary_short'):
            return True, "summary_short changed"
        if backup_entry.get('summary_detailed') != current_entry.get('summary_detailed'):
            return True, "summary_detailed changed"
    
    # For MCQ and Temporal
    else:
        backup_conf = backup_entry.get('confidence', 1.0)
        current_conf = current_entry.get('confidence', 1.0)
        
        if backup_conf != current_conf:
            return True, f"confidence changed: {backup_conf} → {current_conf}"
        
        # Check if question changed
        if task_key == 'mcq':
            if backup_entry.get('question') != current_entry.get('question'):
                return True, "question changed"
            if backup_entry.get('options') != current_entry.get('options'):
                return True, "options changed"
        
        if task_key == 'temporal':
            if backup_entry.get('questions') != current_entry.get('questions'):
                return True, "questions changed"
    
    return False, ""

def create_entry_key(entry: Dict, task_key: str) -> str:
    """Create a unique key for an entry"""
    video_id = entry.get('video_id', 'unknown')
    
    if task_key == 'summary':
        return video_id
    else:
        # For MCQ and Temporal, include segment
        seg = entry.get('segment', {})
        return f"{video_id}_{seg.get('start', 0)}_{seg.get('end', 0)}"

def main():
    backup_path = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa_backup_reviewed/vqa"
    current_path = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa"
    
    print("=" * 80)
    print("VQA BACKUP COMPARISON - Verify Only Failed Entries Were Updated")
    print("=" * 80)
    print(f"\nBackup path:  {backup_path}")
    print(f"Current path: {current_path}\n")
    
    # Load data
    print("Loading backup data...")
    backup_data = load_all_entries(backup_path)
    
    print("Loading current data...")
    current_data = load_all_entries(current_path)
    
    # Statistics
    stats = {
        'summary': {'total': 0, 'changed': 0, 'confidence_only': 0, 'content_changed': 0},
        'mcq': {'total': 0, 'changed': 0, 'confidence_only': 0, 'content_changed': 0},
        'temporal': {'total': 0, 'changed': 0, 'confidence_only': 0, 'content_changed': 0}
    }
    
    changes_detail = []
    
    # Compare each task
    for task_name in ['task1_summarization', 'task2_mcq', 'task3_temporal_localization']:
        task_key = 'summary' if 'task1' in task_name else ('mcq' if 'task2' in task_name else 'temporal')
        
        backup_files = backup_data[task_name]
        current_files = current_data[task_name]
        
        # Compare each file
        for filename in backup_files.keys():
            if filename not in current_files:
                print(f"Warning: {filename} not found in current data")
                continue
            
            backup_entries = backup_files[filename]
            current_entries = current_files[filename]
            
            # Create lookup dicts
            backup_dict = {create_entry_key(e, task_key): e for e in backup_entries}
            current_dict = {create_entry_key(e, task_key): e for e in current_entries}
            
            stats[task_key]['total'] += len(backup_entries)
            
            # Compare entries
            for key, backup_entry in backup_dict.items():
                if key not in current_dict:
                    print(f"Warning: Entry {key} not found in current data")
                    continue
                
                current_entry = current_dict[key]
                
                is_different, reason = compare_entries(backup_entry, current_entry, task_key)
                
                if is_different:
                    stats[task_key]['changed'] += 1
                    
                    # Extract topic info from filename (e.g., "01_Patient-Doctor_Consultations.json")
                    topic_parts = filename.replace('.json', '').split('_', 1)
                    topic_id = topic_parts[0] if len(topic_parts) > 0 else 'unknown'
                    topic_name = topic_parts[1].replace('_', ' ') if len(topic_parts) > 1 else 'unknown'
                    
                    # Get video info
                    video_id = backup_entry.get('video_id', 'unknown')
                    video_number = backup_entry.get('video_number', 'unknown')
                    segment = backup_entry.get('segment', {})
                    
                    # Check if only confidence changed vs content changed
                    if 'confidence changed' in reason and len(reason.split('→')) == 2:
                        # Only confidence changed
                        stats[task_key]['confidence_only'] += 1
                        
                        # Get the confidence values
                        backup_conf = backup_entry.get('confidence', 1.0)
                        current_conf = current_entry.get('confidence', 1.0)
                        
                        changes_detail.append({
                            'task': task_key,
                            'topic_id': topic_id,
                            'topic_name': topic_name,
                            'file': filename,
                            'video_id': video_id,
                            'video_number': video_number,
                            'segment': segment,
                            'key': key,
                            'backup_conf': backup_conf,
                            'current_conf': current_conf,
                            'reason': reason
                        })
                    else:
                        stats[task_key]['content_changed'] += 1
                        changes_detail.append({
                            'task': task_key,
                            'topic_id': topic_id,
                            'topic_name': topic_name,
                            'file': filename,
                            'video_id': video_id,
                            'video_number': video_number,
                            'segment': segment,
                            'key': key,
                            'backup_conf': backup_entry.get('confidence', 1.0),
                            'current_conf': current_entry.get('confidence', 1.0),
                            'reason': reason
                        })
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    for task in ['summary', 'mcq', 'temporal']:
        task_name = {
            'summary': 'Task 1: Summarization',
            'mcq': 'Task 2: MCQ',
            'temporal': 'Task 3: Temporal Localization'
        }[task]
        
        print(f"\n{task_name}")
        print("-" * 80)
        print(f"  Total entries:                  {stats[task]['total']}")
        print(f"  Changed entries:                {stats[task]['changed']}")
        print(f"    - Confidence only changed:    {stats[task]['confidence_only']}")
        print(f"    - Content also changed:       {stats[task]['content_changed']}")
    
    total_changed = sum(s['changed'] for s in stats.values())
    total_conf_only = sum(s['confidence_only'] for s in stats.values())
    total_content = sum(s['content_changed'] for s in stats.values())
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total entries changed:          {total_changed}")
    print(f"  - Confidence only:            {total_conf_only}")
    print(f"  - Content also changed:       {total_content}")
    
    # Expected vs Actual
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    expected_updates = 54  # From your fix_confidence.py run
    
    if total_changed == expected_updates:
        print(f"✅ PASS: Exactly {expected_updates} entries were updated (as expected)")
    elif total_changed < expected_updates:
        print(f"⚠️  WARNING: Only {total_changed} entries were updated (expected {expected_updates})")
    else:
        print(f"❌ FAIL: {total_changed} entries were updated (expected only {expected_updates})")
    
    if total_content > 0:
        print(f"⚠️  WARNING: {total_content} entries had content changes (not just confidence)")
    else:
        print(f"✅ PASS: No content changes, only confidence values were updated")
    
    # Show details of changed entries
    if changes_detail:
        print("\n" + "=" * 80)
        print("CHANGED ENTRIES DETAIL")
        print("=" * 80)
        
        for i, change in enumerate(changes_detail, 1):
            seg_info = ""
            if change.get('segment'):
                seg = change['segment']
                seg_info = f" | Segment: {seg.get('start', 0)}-{seg.get('end', 0)}"
            
            print(f"\n{i}. Task: {change['task'].upper()}")
            print(f"   Topic: [{change['topic_id']}] {change['topic_name']}")
            print(f"   Video ID: {change['video_id']} | Video #: {change['video_number']}{seg_info}")
            print(f"   Confidence: {change['backup_conf']} → {change['current_conf']}")
            print(f"   Change: {change['reason']}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()