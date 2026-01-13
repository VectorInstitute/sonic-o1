import json
import os
import re
from pathlib import Path
import shutil
from datetime import datetime

def detect_failure(entry, task_key):
    """
    Detect if an entry has failed generation
    Returns True if failed, False otherwise
    """
    failed_pattern = re.compile(r'failed\s+to\s+\w+|failed\s+\w+|unable\s+to\s+generate|unable\s+to\s+identify', re.IGNORECASE)
    
    if task_key == 'summary':
        # Check summary fields for failed text or empty/null values
        for field in ['summary_short', 'summary_detailed']:
            field_value = entry.get(field, '')
            # Check if empty, null, or contains failed pattern
            if not field_value or (isinstance(field_value, list) and len(field_value) == 0):
                return True
            if failed_pattern.search(str(field_value)):
                return True
    
    elif task_key == 'mcq':
        # Check question field for failed text or empty/null
        question = entry.get('question', '')
        if not question or question.strip() == '':
            return True
        if failed_pattern.search(str(question)):
            return True
        
        # Check options for "Unable to generate"
        options = entry.get('options', [])
        for option in options:
            if failed_pattern.search(str(option)):
                return True
        
        # Check rationale for failed pattern
        rationale = entry.get('rationale', '')
        if failed_pattern.search(str(rationale)):
            return True
        
        # Check demographics for "failed" value
        demographics = entry.get('demographics', '')
        if demographics == 'failed' or (isinstance(demographics, str) and failed_pattern.search(demographics)):
            return True
    
    elif task_key == 'temporal':
        # Check questions array - if empty or contains failed questions
        questions = entry.get('questions', [])
        if not questions or len(questions) == 0:
            return True
        
        for q in questions:
            q_text = q.get('question', '')
            anchor = q.get('anchor_event', '')
            target = q.get('target_event', '')
            rationale = q.get('rationale_model', '')
            answer = q.get('answer', {})
            
            # Check if question text is empty or contains failed pattern
            if not q_text or q_text.strip() == '' or failed_pattern.search(str(q_text)):
                return True
            
            # Check anchor_event for "Unable to identify"
            if failed_pattern.search(str(anchor)):
                return True
            
            # Check target_event for "Unable to identify"
            if failed_pattern.search(str(target)):
                return True
            
            # Check rationale_model for "Failed to generate"
            if failed_pattern.search(str(rationale)):
                return True
            
            # Check if answer has null values
            if isinstance(answer, dict):
                if answer.get('start_s') is None or answer.get('end_s') is None:
                    return True
            
            # Check abstained flag
            if q.get('abstained', False):
                return True
    
    return False


def update_confidence_for_failed(base_path, dry_run=True, backup=True):
    """
    Update confidence to 0.0 for all failed entries
    
    Args:
        base_path: Path to VQA data directory
        dry_run: If True, only report changes without modifying files
        backup: If True, create backup files before modifying
    """
    
    stats = {
        'summary': {'total': 0, 'updated': 0, 'files': []},
        'mcq': {'total': 0, 'updated': 0, 'files': []},
        'temporal': {'total': 0, 'updated': 0, 'files': []}
    }
    
    # Process each task
    for task_name in ['task1_summarization', 'task2_mcq', 'task3_temporal_localization']:
        task_path = os.path.join(base_path, task_name)
        
        if not os.path.exists(task_path):
            continue
            
        # Get all JSON files in the task directory
        json_files = list(Path(task_path).glob('*.json'))
        
        for json_file in json_files:
            try:
                # Create backup BEFORE reading if we're in apply mode
                backup_created = False
                if not dry_run and backup:
                    backup_path = str(json_file) + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S')
                    shutil.copy2(json_file, backup_path)
                    backup_created = True
                    print(f"  Created backup: {os.path.basename(backup_path)}")
                
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                task_key = 'summary' if 'task1' in task_name else ('mcq' if 'task2' in task_name else 'temporal')
                modified = False
                
                # Process each entry
                for entry in data.get('entries', []):
                    stats[task_key]['total'] += 1
                    
                    # Check if entry has failed but confidence != 0.0
                    if detect_failure(entry, task_key) and entry.get('confidence', 1.0) != 0.0:
                        old_conf = entry.get('confidence', 1.0)
                        
                        if not dry_run:
                            entry['confidence'] = 0.0
                        
                        stats[task_key]['updated'] += 1
                        modified = True
                        
                        # Store info about this update
                        if json_file.name not in [f['file'] for f in stats[task_key]['files']]:
                            stats[task_key]['files'].append({
                                'file': json_file.name,
                                'path': str(json_file),
                                'entries_updated': 1,
                                'examples': [(entry.get('video_id', 'unknown'), old_conf)]
                            })
                        else:
                            for f_info in stats[task_key]['files']:
                                if f_info['file'] == json_file.name:
                                    f_info['entries_updated'] += 1
                                    if len(f_info['examples']) < 3:
                                        f_info['examples'].append((entry.get('video_id', 'unknown'), old_conf))
                
                # Write back modified file
                if modified and not dry_run:
                    # Write updated JSON
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    print(f"  ✓ Updated: {json_file.name}")
                        
            except Exception as e:
                print(f"  ✗ Error processing {json_file}: {e}")
    
    return stats


def print_update_report(stats, dry_run=True):
    """Print report of updates"""
    
    mode = "DRY RUN - NO FILES MODIFIED" if dry_run else "FILES UPDATED"
    
    print("=" * 70)
    print(f"CONFIDENCE UPDATE REPORT - {mode}")
    print("=" * 70)
    print()
    
    total_updated = 0
    
    for task in ['summary', 'mcq', 'temporal']:
        task_name = {
            'summary': 'Task 1: Summarization',
            'mcq': 'Task 2: MCQ',
            'temporal': 'Task 3: Temporal Localization'
        }[task]
        
        print(f"\n{task_name}")
        print("-" * 70)
        print(f"  Total entries checked:      {stats[task]['total']}")
        print(f"  Entries to update:          {stats[task]['updated']}")
        print(f"  Files affected:             {len(stats[task]['files'])}")
        
        if stats[task]['files']:
            print(f"\n  Files with updates:")
            for f_info in stats[task]['files']:
                print(f"    • {f_info['file']}")
                print(f"      - Entries updated: {f_info['entries_updated']}")
                print(f"      - Examples: ", end='')
                for vid, conf in f_info['examples']:
                    print(f"video={vid} (conf {conf}→0.0)", end='; ')
                print()
        
        total_updated += stats[task]['updated']
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total entries to update: {total_updated}")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were modified")
        print("Run with --apply flag to actually update the files")
    else:
        print(f"\n✓ Successfully updated {total_updated} entries")
        print("✓ Backup files created with .backup_YYYYMMDD_HHMMSS extension")
    print()


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update confidence to 0.0 for failed VQA entries'
    )
    parser.add_argument(
        'path',
        help='Path to VQA data directory'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (default is dry-run mode)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files (not recommended)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist!")
        sys.exit(1)
    
    dry_run = not args.apply
    backup = not args.no_backup
    
    print("Analyzing VQA dataset...")
    print(f"Base path: {args.path}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'APPLY CHANGES'}")
    print(f"Backup: {'Yes' if backup and not dry_run else 'No'}\n")
    
    stats = update_confidence_for_failed(args.path, dry_run=dry_run, backup=backup)
    print_update_report(stats, dry_run=dry_run)