import json
import os
from pathlib import Path
from collections import defaultdict

def check_failures(directory_path):
    """
    Check all JSON files for ACTUAL failures using the same detection logic as the main pipeline.
    """
    stats = {
        'total_topics': 0,
        'total_videos': 0,
        'failed_videos': 0,
        'failure_details': defaultdict(list),
        'videos_by_topic': defaultdict(int),
        'failures_by_topic': defaultdict(int)
    }
    
    # Get all JSON files (excluding backups)
    json_files = [f for f in Path(directory_path).glob('*.json') 
                  if not f.name.endswith('.backup_20251123_005808')]
    
    json_files.sort()
    
    print("=" * 80)
    print("VIDEO FAILURE ANALYSIS REPORT (ACTUAL FAILURES ONLY)")
    print("=" * 80)
    print()
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            topic_name = data.get('topic_name', 'Unknown')
            topic_id = data.get('topic_id', 'Unknown')
            entries = data.get('entries', [])
            
            stats['total_topics'] += 1
            stats['total_videos'] += len(entries)
            stats['videos_by_topic'][topic_name] = len(entries)
            
            topic_failures = []
            
            # Check each entry for failures
            for entry in entries:
                video_id = entry.get('video_id', 'Unknown')
                video_number = entry.get('video_number', 'Unknown')
                title = entry.get('title', 'Unknown')
                confidence = entry.get('confidence', 1.0)
                
                # Use EXACT same failure detection logic as the pipeline
                has_summary_failure = False
                failure_reasons = []
                
                # Check summary_short for failures (more specific patterns)
                summary_short = entry.get('summary_short', [])
                if isinstance(summary_short, list):
                    for item in summary_short:
                        if isinstance(item, str):
                            lower_item = item.lower()
                            # Look for specific failure patterns, not just the word "failure"
                            if ('unavailable' in lower_item or 
                                'summary generation failed' in lower_item or
                                'could not be generated' in lower_item or
                                'summary failed' in lower_item or
                                ('first segment' in lower_item and 'failed' in lower_item)):
                                has_summary_failure = True
                                failure_reasons.append(('summary_short', item))
                                break
                
                # Check summary_detailed for failures (more specific patterns)
                summary_detailed = entry.get('summary_detailed', '')
                if isinstance(summary_detailed, str):
                    lower_detailed = summary_detailed.lower()
                    if ('could not be generated' in lower_detailed or 
                        'summary generation failed' in lower_detailed or
                        'parsing error' in lower_detailed or
                        (('failed to' in lower_detailed) and ('summary' in lower_detailed)) or
                        'explicitly reported a failure' in lower_detailed):
                        has_summary_failure = True
                        failure_reasons.append(('summary_detailed', summary_detailed))
                
                # Check confidence
                confidence_failure = confidence <= 0
                if confidence_failure:
                    failure_reasons.append(('confidence', f'Confidence is {confidence}'))
                
                # Determine if this is a real failure
                is_failed = has_summary_failure or confidence_failure
                
                if is_failed:
                    stats['failed_videos'] += 1
                    stats['failures_by_topic'][topic_name] += 1
                    
                    failure_info = {
                        'video_id': video_id,
                        'video_number': video_number,
                        'title': title,
                        'confidence': confidence,
                        'failure_reasons': failure_reasons
                    }
                    topic_failures.append(failure_info)
                    stats['failure_details'][topic_name].append(failure_info)
            
            # Print topic summary
            if topic_failures:
                print(f"📁 Topic {topic_id}: {topic_name}")
                print(f"   Total videos: {len(entries)}, Failed: {len(topic_failures)}")
                print()
                
                for fail in topic_failures:
                    print(f"   ❌ Video {fail['video_number']} - ID: {fail['video_id']}")
                    print(f"      Title: {fail['title']}")
                    print(f"      Confidence: {fail['confidence']:.2f}")
                    print(f"      Failure reasons:")
                    
                    for failure_type, failure_content in fail['failure_reasons']:
                        print(f"         • {failure_type.upper()}:")
                        # Print full content for small messages, truncate for large ones
                        if len(str(failure_content)) <= 150:
                            print(f"            {failure_content}")
                        else:
                            print(f"            {str(failure_content)[:150]}...")
                            print(f"            [Full length: {len(str(failure_content))} chars]")
                    print()
            
        except Exception as e:
            print(f"❌ Error processing {json_file.name}: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
    
    # Print overall summary
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total Topics Processed: {stats['total_topics']}")
    print(f"Total Videos: {stats['total_videos']}")
    print(f"Failed Videos: {stats['failed_videos']}")
    if stats['total_videos'] > 0:
        success_rate = ((stats['total_videos'] - stats['failed_videos']) / stats['total_videos'] * 100)
        failure_rate = (stats['failed_videos'] / stats['total_videos'] * 100)
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Failure Rate: {failure_rate:.2f}%")
    print()
    
    # Topics with failures
    if stats['failures_by_topic']:
        print("Topics with Failures:")
        topic_failure_rates = [
            (topic, fail_count, stats['videos_by_topic'][topic], 
             fail_count / stats['videos_by_topic'][topic] * 100)
            for topic, fail_count in stats['failures_by_topic'].items()
        ]
        topic_failure_rates.sort(key=lambda x: x[3], reverse=True)
        
        for topic, fail_count, total, rate in topic_failure_rates:
            print(f"   • {topic}: {fail_count}/{total} failed ({rate:.1f}%)")
        print()
    
    # Failure type breakdown
    failure_type_counts = defaultdict(int)
    for topic_failures in stats['failure_details'].values():
        for failure in topic_failures:
            for failure_type, _ in failure['failure_reasons']:
                failure_type_counts[failure_type] += 1
    
    if failure_type_counts:
        print("Failure Type Breakdown:")
        for failure_type, count in sorted(failure_type_counts.items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"   • {failure_type}: {count} occurrences")
        print()
    
    # List all failed video IDs for easy reprocessing
    if stats['failed_videos'] > 0:
        print("=" * 80)
        print("FAILED VIDEO IDs (for reprocessing)")
        print("=" * 80)
        all_failed_ids = []
        for topic_failures in stats['failure_details'].values():
            for failure in topic_failures:
                all_failed_ids.append(failure['video_id'])
        
        print(f"Total failed video IDs: {len(all_failed_ids)}")
        print("Video IDs:")
        for vid_id in sorted(all_failed_ids):
            print(f"   {vid_id}")
        print()
    
    return stats

# Run the analysis
directory = "/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/vqa/task1_summarization"
results = check_failures(directory)