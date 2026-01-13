import json
import os
from pathlib import Path
from typing import Dict, List

def load_all_entries(base_path: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Load all entries from VQA directory"""
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

def create_entry_key(entry: Dict, task_key: str) -> str:
    """Create a unique key for an entry"""
    video_id = entry.get('video_id', 'unknown')
    
    if task_key == 'summary':
        return video_id
    else:
        seg = entry.get('segment', {})
        return f"{video_id}_{seg.get('start', 0)}_{seg.get('end', 0)}"

def main():
    backup_path = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa_backup_reviewed/vqa"
    current_path = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa"
    
    print("=" * 80)
    print("FINDING MISSING ENTRIES")
    print("=" * 80)
    print(f"\nBackup path:  {backup_path}")
    print(f"Current path: {current_path}\n")
    
    # Load data
    print("Loading backup data...")
    backup_data = load_all_entries(backup_path)
    
    print("Loading current data...")
    current_data = load_all_entries(current_path)
    
    missing_entries = []
    
    # Check each task
    for task_name in ['task1_summarization', 'task2_mcq', 'task3_temporal_localization']:
        task_key = 'summary' if 'task1' in task_name else ('mcq' if 'task2' in task_name else 'temporal')
        
        backup_files = backup_data[task_name]
        current_files = current_data[task_name]
        
        # Compare each file
        for filename in backup_files.keys():
            if filename not in current_files:
                print(f"⚠️  Warning: File {filename} not found in current data")
                continue
            
            backup_entries = backup_files[filename]
            current_entries = current_files[filename]
            
            # Extract topic info
            topic_parts = filename.replace('.json', '').split('_', 1)
            topic_id = topic_parts[0] if len(topic_parts) > 0 else 'unknown'
            topic_name = topic_parts[1].replace('_', ' ') if len(topic_parts) > 1 else 'unknown'
            
            # Create lookup dicts
            backup_dict = {create_entry_key(e, task_key): e for e in backup_entries}
            current_dict = {create_entry_key(e, task_key): e for e in current_entries}
            
            # Find missing entries
            for key, backup_entry in backup_dict.items():
                if key not in current_dict:
                    video_id = backup_entry.get('video_id', 'unknown')
                    video_number = backup_entry.get('video_number', 'unknown')
                    segment = backup_entry.get('segment', {})
                    confidence = backup_entry.get('confidence', 1.0)
                    
                    missing_entries.append({
                        'task': task_key,
                        'topic_id': topic_id,
                        'topic_name': topic_name,
                        'file': filename,
                        'video_id': video_id,
                        'video_number': video_number,
                        'segment': segment,
                        'key': key,
                        'confidence': confidence
                    })
    
    # Report results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if not missing_entries:
        print("\n✅ No missing entries found - all backup entries exist in current VQA")
    else:
        print(f"\n❌ Found {len(missing_entries)} entries in backup that are MISSING from current VQA\n")
        
        # Group by task
        by_task = {'summary': [], 'mcq': [], 'temporal': []}
        for entry in missing_entries:
            by_task[entry['task']].append(entry)
        
        for task in ['summary', 'mcq', 'temporal']:
            if by_task[task]:
                task_name = {
                    'summary': 'Task 1: Summarization',
                    'mcq': 'Task 2: MCQ',
                    'temporal': 'Task 3: Temporal Localization'
                }[task]
                
                print(f"\n{task_name}: {len(by_task[task])} missing entries")
                print("-" * 80)
                
                for i, entry in enumerate(by_task[task], 1):
                    seg_info = ""
                    if entry.get('segment'):
                        seg = entry['segment']
                        seg_info = f" | Segment: {seg.get('start', 0)}-{seg.get('end', 0)}"
                    
                    print(f"{i}. Topic: [{entry['topic_id']}] {entry['topic_name']}")
                    print(f"   Video ID: {entry['video_id']} | Video #: {entry['video_number']}{seg_info}")
                    print(f"   Confidence in backup: {entry['confidence']}")
                    print(f"   File: {entry['file']}")
                    print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()