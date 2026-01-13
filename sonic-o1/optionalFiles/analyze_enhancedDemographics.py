#!/usr/bin/env python3
"""
Generate CSV reports and plots showing demographic and duration counts from metadata_enhanced.json
Handles normalization of demographic variations and creates visualizations
"""
import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATASET_PATH = "/VideoQA-Agentic/VideoAudioRepDataset/dataset/videos"
OUTPUT_DIR = "/VideoQA-Agentic/VideoAudioRepDataset/optionalFiles/DemographicsAnalysis"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

DEMOGRAPHIC_MAPPING = {
    'race': {
        'white': 'White',
        'caucasian': 'White',
        'european': 'White',
        'black': 'Black',
        'african': 'Black',
        'african american': 'Black',
        'african-american': 'Black',
        'asian': 'Asian',
        'south asian': 'Asian',
        'east asian': 'Asian',
        'southeast asian': 'Asian',
        'east/southeast asian': 'Asian',
        'chinese': 'Asian',
        'japanese': 'Asian',
        'korean': 'Asian',
        'indian': 'Asian',
        'pakistani': 'Asian',
        'bangladeshi': 'Asian',
        'filipino': 'Asian',
        'vietnamese': 'Asian',
        'thai': 'Asian',
        'indigenous': 'Indigenous',
        'native': 'Indigenous',
        'native american': 'Indigenous',
        'first nations': 'Indigenous',
        'aboriginal': 'Indigenous',
        'arab': 'Arab',
        'middle eastern': 'Arab',
        'middle-eastern': 'Arab',
        'hispanic': 'Hispanic',
        'latino': 'Hispanic',
        'latina': 'Hispanic',
        'latinx': 'Hispanic',
        'latin american': 'Hispanic',
        'mexican': 'Hispanic',
        'mixed': None,
        'mixed race': None,
        'biracial': None,
        'multiracial': None,
        'pacific islander': None,
    },
    'gender': {
        'male': 'Male',
        'female': 'Female',
        'man': 'Male',
        'woman': 'Female',
        'men': 'Male',
        'women': 'Female',
        'boy': 'Male',
        'girl': 'Female',
        'masculine': 'Male',
        'feminine': 'Female',
    },
    'age': {
        'young': 'Young (18-24)',
        'young (18-24)': 'Young (18-24)',
        'young (18–24)': 'Young (18-24)',
        'young adult': 'Young (18-24)',
        'youth': 'Young (18-24)',
        'teenage': 'Young (18-24)',
        'teen': 'Young (18-24)',
        'adolescent': 'Young (18-24)',
        'young (18-34)': 'Young (18-24)',
        'middle': 'Middle (25-39)',
        'middle (25-39)': 'Middle (25-39)',
        'middle (25–39)': 'Middle (25-39)',
        'middle-aged': 'Middle (25-39)',
        'middle aged': 'Middle (25-39)',
        'adult': 'Middle (25-39)',
        'older': 'Older adults (40+)',
        'older adults': 'Older adults (40+)',
        'older adults (40+)': 'Older adults (40+)',
        'older adult': 'Older adults (40+)',
        'senior': 'Older adults (40+)',
        'elderly': 'Older adults (40+)',
        'mature': 'Older adults (40+)',
        'child': None,
        'children': None,
        'young (children)': None,
        'young (under 12)': None,
        'kid': None,
        'kids': None,
        'infant': None,
        'baby': None,
        'toddler': None,
    },
    'language': {
        'english': 'English',
        'english (american)': 'English',
        'english (british)': 'English',
        'english (australian)': 'English',
        'english (uk)': 'English',
        'english (us)': 'English',
        'american english': 'English',
        'british english': 'English',
        'hindi': 'Hindi',
        'arabic': 'Arabic',
        'spanish': 'Spanish',
        'español': 'Spanish',
        'castilian': 'Spanish',
        'chinese': 'Chinese',
        'mandarin': 'Chinese',
        'cantonese': 'Chinese',
        'mandarin chinese': 'Chinese',
    }
}


def normalize_demographic_value(category, value):
    """Normalize demographic value to canonical form, return None for excluded values"""
    if not value or not isinstance(value, str):
        return None
    
    value_lower = value.lower().strip()
    
    if category in DEMOGRAPHIC_MAPPING:
        if value_lower in DEMOGRAPHIC_MAPPING[category]:
            return DEMOGRAPHIC_MAPPING[category][value_lower]
        
        if category == 'language':
            return None
        
        return value.title()
    
    return value.title()


def load_enhanced_metadata(base_path):
    """Load all metadata_enhanced.json files from topic folders"""
    all_metadata = {}
    base = Path(base_path)
    
    if not base.exists():
        return all_metadata
    
    for topic_folder in sorted(base.iterdir()):
        if topic_folder.is_dir():
            metadata_file = topic_folder / "metadata_enhanced.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        all_metadata[topic_folder.name] = data
                except Exception as e:
                    print(f"Error loading {topic_folder.name}: {e}")
    
    return all_metadata


def extract_demographics_from_videos(videos):
    """Extract demographic counts from video metadata with normalization"""
    race_counter = Counter()
    gender_counter = Counter()
    age_counter = Counter()
    language_counter = Counter()
    duration_counter = Counter()
    
    total_videos = len(videos)
    videos_with_captions = 0
    total_duration_minutes = 0
    
    for video in videos:
        duration_seconds = video.get('duration_seconds', 0)
        total_duration_minutes += duration_seconds / 60
        duration_category = video.get('duration_category', 'unknown')
        duration_counter[duration_category] += 1
        
        if video.get('has_captions', False):
            videos_with_captions += 1
        
        demographics_detailed = video.get('demographics_detailed', {})
        
        for race in demographics_detailed.get('race', []):
            normalized = normalize_demographic_value('race', race)
            if normalized:
                race_counter[normalized] += 1
        
        for gender in demographics_detailed.get('gender', []):
            normalized = normalize_demographic_value('gender', gender)
            if normalized:
                gender_counter[normalized] += 1
        
        for age in demographics_detailed.get('age', []):
            normalized = normalize_demographic_value('age', age)
            if normalized:
                age_counter[normalized] += 1
        
        for language in demographics_detailed.get('language', []):
            normalized = normalize_demographic_value('language', language)
            if normalized:
                language_counter[normalized] += 1
    
    return {
        'total_videos': total_videos,
        'videos_with_captions': videos_with_captions,
        'total_duration_minutes': total_duration_minutes,
        'avg_duration_minutes': total_duration_minutes / total_videos if total_videos > 0 else 0,
        'race': dict(race_counter),
        'gender': dict(gender_counter),
        'age': dict(age_counter),
        'language': dict(language_counter),
        'duration': dict(duration_counter)
    }


def create_demographics_csv(all_metadata):
    """Create CSV with demographic counts for each topic"""
    all_races = set()
    all_genders = set()
    all_ages = set()
    all_languages = set()
    
    topic_stats = {}
    
    for topic_id, videos in all_metadata.items():
        stats = extract_demographics_from_videos(videos)
        topic_stats[topic_id] = stats
        
        all_races.update(stats['race'].keys())
        all_genders.update(stats['gender'].keys())
        all_ages.update(stats['age'].keys())
        all_languages.update(stats['language'].keys())
    
    race_columns = sorted(all_races)
    gender_columns = sorted(all_genders)
    age_columns = sorted(all_ages)
    language_columns = sorted(all_languages)
    
    rows = []
    for topic_id, stats in sorted(topic_stats.items()):
        videos = all_metadata[topic_id]
        topic_name = videos[0].get('topic_name', topic_id) if videos else topic_id
        
        row = {
            'Topic_ID': topic_id,
            'Topic_Name': topic_name,
            'Total_Videos': stats['total_videos']
        }
        
        for race in race_columns:
            row[f'Race_{race}'] = stats['race'].get(race, 0)
        
        for gender in gender_columns:
            row[f'Gender_{gender}'] = stats['gender'].get(gender, 0)
        
        for age in age_columns:
            row[f'Age_{age}'] = stats['age'].get(age, 0)
        
        for language in language_columns:
            row[f'Language_{language}'] = stats['language'].get(language, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    total_row = {'Topic_ID': 'TOTAL', 'Topic_Name': 'ALL TOPICS', 'Total_Videos': df['Total_Videos'].sum()}
    for col in df.columns:
        if col not in ['Topic_ID', 'Topic_Name', 'Total_Videos']:
            total_row[col] = df[col].sum()
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'demographics_detailed_by_topic.csv')
    df.to_csv(output_path, index=False)
    
    return df, topic_stats


def create_duration_csv(all_metadata):
    """Create CSV with duration counts for each topic"""
    all_durations = set()
    topic_stats = {}
    
    for topic_id, videos in all_metadata.items():
        stats = extract_demographics_from_videos(videos)
        topic_stats[topic_id] = stats
        all_durations.update(stats['duration'].keys())
    
    duration_columns = sorted(all_durations)
    
    rows = []
    for topic_id, stats in sorted(topic_stats.items()):
        videos = all_metadata[topic_id]
        topic_name = videos[0].get('topic_name', topic_id) if videos else topic_id
        
        row = {
            'Topic_ID': topic_id,
            'Topic_Name': topic_name,
            'Total_Videos': stats['total_videos'],
            'Total_Duration_Min': round(stats['total_duration_minutes'], 2),
            'Avg_Duration_Min': round(stats['avg_duration_minutes'], 2)
        }
        
        for dur in duration_columns:
            row[f'Duration_{dur}'] = stats['duration'].get(dur, 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    total_row = {
        'Topic_ID': 'TOTAL',
        'Topic_Name': 'ALL TOPICS',
        'Total_Videos': df['Total_Videos'].sum(),
        'Total_Duration_Min': df['Total_Duration_Min'].sum(),
        'Avg_Duration_Min': round(df['Total_Duration_Min'].sum() / df['Total_Videos'].sum(), 2)
    }
    for col in df.columns:
        if col.startswith('Duration_'):
            total_row[col] = df[col].sum()
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'duration_by_topic.csv')
    df.to_csv(output_path, index=False)
    
    return df


def create_summary_stats_csv(all_metadata):
    """Create summary statistics CSV"""
    rows = []
    
    for topic_id, videos in sorted(all_metadata.items()):
        stats = extract_demographics_from_videos(videos)
        topic_name = videos[0].get('topic_name', topic_id) if videos else topic_id
        
        row = {
            'Topic_ID': topic_id,
            'Topic_Name': topic_name,
            'Total_Videos': stats['total_videos'],
            'Videos_With_Captions': stats['videos_with_captions'],
            'Videos_Without_Captions': stats['total_videos'] - stats['videos_with_captions'],
            'Total_Duration_Minutes': round(stats['total_duration_minutes'], 2),
            'Avg_Duration_Minutes': round(stats['avg_duration_minutes'], 2),
            'Unique_Races': len(stats['race']),
            'Unique_Genders': len(stats['gender']),
            'Unique_Age_Groups': len(stats['age']),
            'Unique_Languages': len(stats['language']),
            'Total_Race_Mentions': sum(stats['race'].values()),
            'Total_Gender_Mentions': sum(stats['gender'].values()),
            'Total_Age_Mentions': sum(stats['age'].values()),
            'Total_Language_Mentions': sum(stats['language'].values())
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    total_row = {
        'Topic_ID': 'TOTAL',
        'Topic_Name': 'ALL TOPICS',
        'Total_Videos': df['Total_Videos'].sum(),
        'Videos_With_Captions': df['Videos_With_Captions'].sum(),
        'Videos_Without_Captions': df['Videos_Without_Captions'].sum(),
        'Total_Duration_Minutes': df['Total_Duration_Minutes'].sum(),
        'Avg_Duration_Minutes': round(df['Total_Duration_Minutes'].sum() / df['Total_Videos'].sum(), 2),
        'Unique_Races': '-',
        'Unique_Genders': '-',
        'Unique_Age_Groups': '-',
        'Unique_Languages': '-',
        'Total_Race_Mentions': df['Total_Race_Mentions'].sum(),
        'Total_Gender_Mentions': df['Total_Gender_Mentions'].sum(),
        'Total_Age_Mentions': df['Total_Age_Mentions'].sum(),
        'Total_Language_Mentions': df['Total_Language_Mentions'].sum()
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'summary_statistics.csv')
    df.to_csv(output_path, index=False)
    
    return df


def create_overall_demographics_csv(all_metadata):
    """Create overall demographic distribution across all topics"""
    overall_race = Counter()
    overall_gender = Counter()
    overall_age = Counter()
    overall_language = Counter()
    
    for videos in all_metadata.values():
        stats = extract_demographics_from_videos(videos)
        overall_race.update(stats['race'])
        overall_gender.update(stats['gender'])
        overall_age.update(stats['age'])
        overall_language.update(stats['language'])
    
    race_df = pd.DataFrame([
        {'Category': 'Race', 'Value': k, 'Count': v, 'Percentage': round(v / sum(overall_race.values()) * 100, 2)}
        for k, v in sorted(overall_race.items(), key=lambda x: x[1], reverse=True)
    ])
    
    gender_df = pd.DataFrame([
        {'Category': 'Gender', 'Value': k, 'Count': v, 'Percentage': round(v / sum(overall_gender.values()) * 100, 2)}
        for k, v in sorted(overall_gender.items(), key=lambda x: x[1], reverse=True)
    ])
    
    age_df = pd.DataFrame([
        {'Category': 'Age', 'Value': k, 'Count': v, 'Percentage': round(v / sum(overall_age.values()) * 100, 2)}
        for k, v in sorted(overall_age.items(), key=lambda x: x[1], reverse=True)
    ])
    
    language_df = pd.DataFrame([
        {'Category': 'Language', 'Value': k, 'Count': v, 'Percentage': round(v / sum(overall_language.values()) * 100, 2)}
        for k, v in sorted(overall_language.items(), key=lambda x: x[1], reverse=True)
    ])
    
    df = pd.concat([race_df, gender_df, age_df, language_df], ignore_index=True)
    
    output_path = os.path.join(OUTPUT_DIR, 'overall_demographics_distribution.csv')
    df.to_csv(output_path, index=False)
    
    return df, {
        'race': dict(overall_race),
        'gender': dict(overall_gender),
        'age': dict(overall_age),
        'language': dict(overall_language)
    }


def create_overall_plots(overall_stats):
    """Create overall demographic distribution plots"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Overall Demographic Distribution Across All Topics', fontsize=16, fontweight='bold')
    
    categories = ['race', 'gender', 'age', 'language']
    titles = ['Race Distribution', 'Gender Distribution', 'Age Distribution', 'Language Distribution']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, (category, title, color) in enumerate(zip(categories, titles, colors)):
        ax = axes[idx // 2, idx % 2]
        
        data = overall_stats[category]
        if not data:
            continue
        
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        total = sum(values)
        percentages = [v/total*100 for v in values]
        
        bars = ax.bar(range(len(labels)), values, color=color, alpha=0.7, edgecolor='black')
        
        for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(PLOTS_DIR, 'overall_demographics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_topic_plots(topic_stats, all_metadata):
    """Create demographic distribution plots for each topic"""
    sns.set_style("whitegrid")
    
    for topic_id, stats in sorted(topic_stats.items()):
        videos = all_metadata[topic_id]
        topic_name = videos[0].get('topic_name', topic_id) if videos else topic_id
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Demographics for Topic: {topic_name} (ID: {topic_id})', 
                    fontsize=16, fontweight='bold')
        
        categories = ['race', 'gender', 'age', 'language']
        titles = ['Race Distribution', 'Gender Distribution', 'Age Distribution', 'Language Distribution']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, (category, title, color) in enumerate(zip(categories, titles, colors)):
            ax = axes[idx // 2, idx % 2]
            
            data = stats[category]
            if not data:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                continue
            
            sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_data]
            values = [item[1] for item in sorted_data]
            total = sum(values)
            percentages = [v/total*100 for v in values]
            
            bars = ax.bar(range(len(labels)), values, color=color, alpha=0.7, edgecolor='black')
            
            for i, (bar, val, pct) in enumerate(zip(bars, values, percentages)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val}\n({pct:.1f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Category', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(PLOTS_DIR, f'topic_{topic_id}_demographics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def create_comparison_heatmap(topic_stats, all_metadata):
    """Create heatmap comparing demographics across topics"""
    categories = ['race', 'gender', 'age', 'language']
    
    for category in categories:
        all_values = set()
        for stats in topic_stats.values():
            all_values.update(stats[category].keys())
        
        if not all_values:
            continue
        
        all_values = sorted(all_values)
        
        topic_ids = sorted(topic_stats.keys())
        matrix = []
        topic_names = []
        
        for topic_id in topic_ids:
            videos = all_metadata[topic_id]
            topic_name = videos[0].get('topic_name', topic_id) if videos else topic_id
            topic_names.append(f"{topic_id}\n{topic_name[:20]}")
            
            row = [topic_stats[topic_id][category].get(val, 0) for val in all_values]
            matrix.append(row)
        
        plt.figure(figsize=(max(12, len(all_values)*0.8), max(8, len(topic_ids)*0.5)))
        sns.heatmap(matrix, 
                   xticklabels=all_values,
                   yticklabels=topic_names,
                   annot=True,
                   fmt='d',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Count'},
                   linewidths=0.5)
        
        plt.title(f'{category.title()} Distribution Across Topics', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(category.title(), fontsize=12, fontweight='bold')
        plt.ylabel('Topics', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(PLOTS_DIR, f'heatmap_{category}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def print_summary_report(all_metadata):
    """Print a formatted summary report to console"""
    print("\n" + "="*100)
    print("ENHANCED METADATA ANALYSIS SUMMARY")
    print("="*100)
    
    total_videos = sum(len(videos) for videos in all_metadata.values())
    total_topics = len(all_metadata)
    
    print(f"\nTotal Topics: {total_topics}")
    print(f"Total Videos: {total_videos}")
    print(f"Average Videos per Topic: {total_videos / total_topics:.1f}")
    
    overall_stats = extract_demographics_from_videos(
        [video for videos in all_metadata.values() for video in videos]
    )
    
    print(f"\nTotal Duration: {overall_stats['total_duration_minutes']:.1f} minutes ({overall_stats['total_duration_minutes']/60:.1f} hours)")
    print(f"Average Video Duration: {overall_stats['avg_duration_minutes']:.1f} minutes")
    print(f"Videos with Captions: {overall_stats['videos_with_captions']} ({overall_stats['videos_with_captions']/total_videos*100:.1f}%)")
    
    print("\n" + "-"*100)
    print("DEMOGRAPHIC BREAKDOWN (Total Mentions - Normalized)")
    print("-"*100)
    
    print(f"\nRace Distribution:")
    for race, count in sorted(overall_stats['race'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {race:25s}: {count:4d} ({count/sum(overall_stats['race'].values())*100:5.1f}%)")
    
    print(f"\nGender Distribution:")
    for gender, count in sorted(overall_stats['gender'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {gender:25s}: {count:4d} ({count/sum(overall_stats['gender'].values())*100:5.1f}%)")
    
    print(f"\nAge Distribution:")
    for age, count in sorted(overall_stats['age'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {age:25s}: {count:4d} ({count/sum(overall_stats['age'].values())*100:5.1f}%)")
    
    print(f"\nLanguage Distribution:")
    for language, count in sorted(overall_stats['language'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {language:25s}: {count:4d} ({count/sum(overall_stats['language'].values())*100:5.1f}%)")
    
    print("\n" + "="*100 + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_metadata = load_enhanced_metadata(DATASET_PATH)
    
    if not all_metadata:
        print(f"No metadata files found in: {DATASET_PATH}")
        return
    
    total_videos = sum(len(videos) for videos in all_metadata.values())
    print(f"Loaded {len(all_metadata)} topics with {total_videos} total videos")
    
    demographics_df, topic_stats = create_demographics_csv(all_metadata)
    create_duration_csv(all_metadata)
    create_summary_stats_csv(all_metadata)
    overall_df, overall_stats = create_overall_demographics_csv(all_metadata)
    
    create_overall_plots(overall_stats)
    create_topic_plots(topic_stats, all_metadata)
    create_comparison_heatmap(topic_stats, all_metadata)
    
    print_summary_report(all_metadata)
    
    print(f"\nAnalysis complete. Files saved to: {OUTPUT_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()