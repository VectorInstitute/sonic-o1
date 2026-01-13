import json
import os
from pathlib import Path


def copy_new_demographics_to_extended():
    """
    Copy videos with specific demographics from Unfiltered to QualityAnnotated
    in a new 'extended' directory for review.
    
    To add different demographics, modify the target_demographics list below.
    Format: "category:value" (e.g., "race:Arab", "race:Asian", "gender:Female")
    """
    base_dir = Path(".")
    
    unfiltered_dir = base_dir / "videos_Unfiltered"
    quality_dir = base_dir / "videos_QualityAnnotated"
    extended_dir = base_dir / "videos_QualityAnnotated_Extended"
    
    extended_dir.mkdir(exist_ok=True)
    
    # Configure target demographics here - modify this list to filter different demographics
    # Format: "category:value" where category is race/gender/age/language
    # Examples: "race:Arab", "race:Indigenous", "race:Asian", "gender:Female", "age:Young (18-24)"
    target_demographics = ["race:Arab", "race:Indigenous"]
    
    total_new = 0
    demographic_counts = {demo: 0 for demo in target_demographics}
    
    for topic_folder in sorted(quality_dir.iterdir()):
        if not topic_folder.is_dir():
            continue
            
        topic_name = topic_folder.name
        
        quality_json_files = list(topic_folder.glob("*_metadata.json"))
        unfiltered_topic = unfiltered_dir / topic_name
        
        if not unfiltered_topic.exists():
            continue
            
        unfiltered_json_files = list(unfiltered_topic.glob("*_metadata.json"))
        
        if not quality_json_files or not unfiltered_json_files:
            continue
        
        quality_path = quality_json_files[0]
        unfiltered_path = unfiltered_json_files[0]
        
        with open(quality_path, 'r', encoding='utf-8') as f:
            quality_videos = json.load(f)
        
        with open(unfiltered_path, 'r', encoding='utf-8') as f:
            unfiltered_videos = json.load(f)
        
        existing_ids = {v['video_id'] for v in quality_videos}
        
        new_videos = [
            v for v in unfiltered_videos 
            if v.get('demographic_label') in target_demographics 
            and v['video_id'] not in existing_ids
        ]
        
        if new_videos:
            # Count each demographic separately
            for demo in target_demographics:
                count = sum(1 for v in new_videos if v.get('demographic_label') == demo)
                demographic_counts[demo] += count
            
            extended_videos = quality_videos + new_videos
            
            extended_topic_dir = extended_dir / topic_name
            extended_topic_dir.mkdir(exist_ok=True)
            
            json_filename = quality_path.name
            extended_json_path = extended_topic_dir / json_filename
            
            with open(extended_json_path, 'w', encoding='utf-8') as f:
                json.dump(extended_videos, f, indent=2)
            
            # Create summary with demographic breakdown
            demographic_breakdown = {
                demo: sum(1 for v in new_videos if v.get('demographic_label') == demo)
                for demo in target_demographics
            }
            
            summary = {
                "topic_name": topic_name,
                "original_count": len(quality_videos),
                "new_videos_added": len(new_videos),
                "demographic_breakdown": demographic_breakdown,
                "total_count": len(extended_videos),
                "new_video_ids": [v['video_id'] for v in new_videos]
            }
            
            summary_path = extended_topic_dir / f"extension_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            total_new += len(new_videos)
    
    print(f"Total new videos added: {total_new}")
    for demo, count in demographic_counts.items():
        print(f"  {demo}: {count}")
    print(f"Extended datasets saved to: {extended_dir}/")


if __name__ == "__main__":
    copy_new_demographics_to_extended()