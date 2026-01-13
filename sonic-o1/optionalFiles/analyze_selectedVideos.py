#!/usr/bin/env python3
"""
Generate CSV reports showing demographic and duration counts for each topic
"""

import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Configuration
DATASET_PATH = "/VideoQA-Agentic/VideoAudioRepDataset/dataset/videos"
OUTPUT_DIR = "/VideoQA-Agentic/VideoAudioRepDataset/VideoAudioDemographicsAnalysis"

def load_summaries(base_path):
    """Load all summary.json files from topic folders"""
    summaries = {}
    base = Path(base_path)
    
    if not base.exists():
        print(f"Path not found: {base_path}")
        return summaries
    
    for topic_folder in sorted(base.iterdir()):
        if topic_folder.is_dir():
            summary_file = topic_folder / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                        summaries[topic_folder.name] = data
                        print(f"✓ Loaded: {topic_folder.name}")
                except Exception as e:
                    print(f"✗ Error loading {topic_folder.name}: {e}")
    
    return summaries

def create_demographics_csv(summaries):
    """Create CSV with demographic counts for each topic"""
    
    # Collect all unique demographic categories
    all_demographics = set()
    for data in summaries.values():
        demographics = data.get('distribution', {}).get('by_demographics', {})
        all_demographics.update(demographics.keys())
    
    # Sort demographics for consistent column order
    demographic_columns = sorted(all_demographics)
    
    # Build data rows
    rows = []
    for topic_id, data in sorted(summaries.items()):
        topic_name = data.get('topic_name', topic_id)
        demographics = data.get('distribution', {}).get('by_demographics', {})
        total_videos = data.get('statistics', {}).get('selected_videos_count', 0)
        
        row = {
            'Topic': topic_name,
            'Total_Videos': total_videos
        }
        
        # Add demographic counts
        for demo in demographic_columns:
            row[demo] = demographics.get(demo, 0)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add total row
    total_row = {'Topic': 'TOTAL', 'Total_Videos': df['Total_Videos'].sum()}
    for demo in demographic_columns:
        total_row[demo] = df[demo].sum()
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'demographics_by_topic.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDemographics CSV saved: {output_path}")
    
    return df

def create_duration_csv(summaries):
    """Create CSV with duration counts for each topic"""
    
    # Collect all unique duration categories
    all_durations = set()
    for data in summaries.values():
        durations = data.get('distribution', {}).get('by_duration', {})
        all_durations.update(durations.keys())
    
    # Sort durations for consistent column order
    duration_columns = sorted(all_durations)
    
    # Build data rows
    rows = []
    for topic_id, data in sorted(summaries.items()):
        topic_name = data.get('topic_name', topic_id)
        durations = data.get('distribution', {}).get('by_duration', {})
        total_videos = data.get('statistics', {}).get('selected_videos_count', 0)
        
        row = {
            'Topic': topic_name,
            'Total_Videos': total_videos
        }
        
        # Add duration counts
        for dur in duration_columns:
            row[dur] = durations.get(dur, 0)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Add total row
    total_row = {'Topic': 'TOTAL', 'Total_Videos': df['Total_Videos'].sum()}
    for dur in duration_columns:
        total_row[dur] = df[dur].sum()
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'duration_by_topic.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Duration CSV saved: {output_path}")
    
    return df

def create_combined_csv(summaries):
    """Create comprehensive CSV with all metrics"""
    
    rows = []
    
    for topic_id, data in sorted(summaries.items()):
        topic_name = data.get('topic_name', topic_id)
        stats = data.get('statistics', {})
        demographics = data.get('distribution', {}).get('by_demographics', {})
        durations = data.get('distribution', {}).get('by_duration', {})
        
        row = {
            'Topic': topic_name,
            'Total_Videos': stats.get('selected_videos_count', 0),
            'Total_Duration_Minutes': stats.get('total_duration_minutes', 0),
            'Avg_Duration_Minutes': stats.get('average_duration_minutes', 0),
            'Videos_With_Captions': stats.get('videos_with_captions', 0),
            'Videos_Needing_Whisper': stats.get('videos_needing_whisper', 0),
        }
        
        # Add demographics
        for demo, count in demographics.items():
            row[f'Demo_{demo}'] = count
        
        # Add durations
        for dur, count in durations.items():
            row[f'Dur_{dur}'] = count
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Fill NaN with 0 for missing categories
    df = df.fillna(0)
    
    # Add total row for numeric columns
    total_row = {'Topic': 'TOTAL'}
    for col in df.columns:
        if col != 'Topic' and col != 'Avg_Duration_Minutes':
            total_row[col] = df[col].sum()
        elif col == 'Avg_Duration_Minutes':
            total_row[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'complete_summary.csv')
    df.to_csv(output_path, index=False)
    print(f"✓ Complete summary CSV saved: {output_path}")
    
    return df

def create_percentages_csv(summaries):
    """Create CSV with demographic percentages for each topic"""
    
    # Collect all unique demographic categories
    all_demographics = set()
    for data in summaries.values():
        demographics = data.get('demographic_percentages', {})
        all_demographics.update(demographics.keys())
    
    demographic_columns = sorted(all_demographics)
    
    # Build data rows
    rows = []
    for topic_id, data in sorted(summaries.items()):
        topic_name = data.get('topic_name', topic_id)
        demographics = data.get('demographic_percentages', {})
        total_videos = data.get('statistics', {}).get('selected_videos_count', 0)
        
        row = {
            'Topic': topic_name,
            'Total_Videos': total_videos
        }
        
        # Add demographic percentages
        for demo in demographic_columns:
            row[f'{demo}_%'] = demographics.get(demo, 0)
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'demographics_percentages.csv')
    df.to_csv(output_path, index=False)
    print(f"Demographics percentages CSV saved: {output_path}")
    
    return df

def print_summary_tables(demo_df, dur_df):
    """Print formatted tables to console"""
    
    print("\n" + "="*100)
    print("DEMOGRAPHIC DISTRIBUTION BY TOPIC")
    print("="*100)
    print(demo_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("DURATION DISTRIBUTION BY TOPIC")
    print("="*100)
    print(dur_df.to_string(index=False))
    print("="*100 + "\n")

def main():
    print("Creating CSV Reports for Video Demographics\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load summaries
    summaries = load_summaries(DATASET_PATH)
    
    if not summaries:
        print("\n⚠️  No summaries found. Please check the path.")
        print(f"Looking in: {DATASET_PATH}")
        return
    
    print(f"\n✓ Loaded {len(summaries)} topic summaries\n")
    
    # Generate CSVs
    print("Generating CSV files...\n")
    
    demo_df = create_demographics_csv(summaries)
    dur_df = create_duration_csv(summaries)
    create_percentages_csv(summaries)
    create_combined_csv(summaries)
    
    # Print tables
    print_summary_tables(demo_df, dur_df)
    
    print(f"CSV reports generated successfully!")
    print(f"All files saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  • demographics_by_topic.csv       - Count of videos per demographic")
    print(f"  • demographics_percentages.csv    - Percentage distribution")
    print(f"  • duration_by_topic.csv          - Count of videos per duration category")
    print(f"  • complete_summary.csv           - All metrics in one file")

if __name__ == "__main__":
    main()