import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter

# Set style for research paper quality
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Base path
base_path = Path('/projects/aixpert/users/ahmadradw/VideoQA-Agentic/VideoAudioRepDataset/dataset/videos')

# Get all topic folders
topic_folders = sorted([f for f in base_path.iterdir() if f.is_dir() and f.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_', '12_', '13_'))])

print(f"Found {len(topic_folders)} topic folders")

# Load data from all topics
all_data = []
for topic_folder in topic_folders:
    metadata_file = topic_folder / 'metadata_enhanced.json'
    
    if metadata_file.exists():
        print(f"Loading: {topic_folder.name}")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        for entry in metadata:
            # Extract demographics from demographics_detailed_reviewed
            demo_reviewed = entry.get('demographics_detailed_reviewed', {})
            
            # Get topic name and rename if needed
            topic_name = entry.get('topic_name', topic_folder.name)
            if 'Transportation' in topic_name or topic_name == 'Public Transportation Conflicts':
                topic_name = 'Public Conflicts'
            
            all_data.append({
                'topic_name': topic_name,
                'duration_category': entry['duration_category'],
                'duration_seconds': entry['duration_seconds'],
                'duration_minutes': entry['duration_seconds'] / 60,
                'topic_folder': topic_folder.name,
                'race': demo_reviewed.get('race', []),
                'gender': demo_reviewed.get('gender', []),
                'age': demo_reviewed.get('age', []),
                'language': demo_reviewed.get('language', [])
            })
    else:
        print(f"Warning: No metadata_enhanced.json found in {topic_folder.name}")

df = pd.DataFrame(all_data)
print(f"\nTotal videos loaded: {len(df)}")

# Define colors
colors_main = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
duration_colors = ['#8DD3C7', '#FFFFB3', '#FB8072']
duration_order = ['short', 'medium', 'long']

# ========== FIGURE 1: Number of Videos per Topic ==========
fig1, ax1 = plt.subplots(figsize=(7, 4))

topic_counts = df.groupby('topic_name').size().sort_values(ascending=False)
bars1 = ax1.bar(range(len(topic_counts)), topic_counts.values, 
                color=colors_main[:len(topic_counts)], edgecolor='black', linewidth=0.8)

ax1.set_xlabel('Topic Category')
ax1.set_ylabel('Number of Videos')
ax1.set_xticks(range(len(topic_counts)))
ax1.set_xticklabels(topic_counts.index, rotation=45, ha='right', fontsize=8.5)
ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars1, topic_counts.values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(int(val)), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(base_path.parent / 'fig1_videos_per_topic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(base_path.parent / 'fig1_videos_per_topic.pdf', bbox_inches='tight')
print("\nFigure 1 saved: Videos per topic")
plt.close()

# ========== FIGURE 2: Total Hours per Topic ==========
fig2, ax2 = plt.subplots(figsize=(7, 4))

topic_hours = (df.groupby('topic_name')['duration_seconds'].sum() / 3600).sort_values(ascending=False)
bars2 = ax2.bar(range(len(topic_hours)), topic_hours.values,
                color=colors_main[:len(topic_hours)], edgecolor='black', linewidth=0.8)

ax2.set_xlabel('Topic Category')
ax2.set_ylabel('Total Hours')
ax2.set_xticks(range(len(topic_hours)))
ax2.set_xticklabels(topic_hours.index, rotation=45, ha='right', fontsize=8.5)
ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars2, topic_hours.values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(base_path.parent / 'fig2_hours_per_topic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(base_path.parent / 'fig2_hours_per_topic.pdf', bbox_inches='tight')
print("Figure 2 saved: Total hours per topic")
plt.close()

# ========== FIGURE 3: Distribution Across YOUR 3 Duration Categories ==========
fig3, ax3 = plt.subplots(figsize=(7, 4))

# Use YOUR actual 3 duration categories
duration_cat_counts = df['duration_category'].value_counts().reindex(duration_order, fill_value=0)

bars3 = ax3.bar(range(len(duration_cat_counts)), duration_cat_counts.values,
                color=duration_colors, edgecolor='black', linewidth=0.8)

ax3.set_xlabel('Duration Category')
ax3.set_ylabel('Number of Videos')
ax3.set_xticks(range(len(duration_cat_counts)))
ax3.set_xticklabels(['Short\n(<5 min)', 'Medium\n(5-15 min)', 'Long\n(>15 min)'], fontsize=9)
ax3.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax3.set_axisbelow(True)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars3, duration_cat_counts.values)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(int(val)), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(base_path.parent / 'fig3_duration_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(base_path.parent / 'fig3_duration_distribution.pdf', bbox_inches='tight')
print("Figure 3 saved: Duration distribution (3 categories)")
plt.close()

# ========== FIGURE 4: Stacked Bar - Duration by Topic ==========
fig4, ax4 = plt.subplots(figsize=(8, 6))

# Calculate counts for stacked bar
topic_duration_counts = df.groupby(['topic_name', 'duration_category']).size().unstack(fill_value=0)
topic_duration_counts = topic_duration_counts.reindex(columns=duration_order, fill_value=0)
topic_duration_counts = topic_duration_counts.loc[topic_duration_counts.sum(axis=1).sort_values(ascending=True).index]

# Create stacked horizontal bar
x = np.arange(len(topic_duration_counts))
width = 0.7

# Plot each duration category
bottom = np.zeros(len(topic_duration_counts))
for i, duration_cat in enumerate(duration_order):
    values = topic_duration_counts[duration_cat].values
    bars = ax4.barh(x, values, width, left=bottom, label=duration_cat.capitalize(),
                    color=duration_colors[i], edgecolor='white', linewidth=0.5)
    
    # Add value labels (only if > 0)
    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 0:
            ax4.text(bottom[j] + val/2, bar.get_y() + bar.get_height()/2,
                    str(int(val)), ha='center', va='center', fontsize=8)
    
    bottom += values

ax4.set_yticks(x)
ax4.set_yticklabels(topic_duration_counts.index, fontsize=8.5)
ax4.set_xlabel('Number of Videos')
ax4.set_ylabel('Topic')
ax4.legend(title='Duration', loc='lower right', frameon=True, fontsize=9)
ax4.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax4.set_axisbelow(True)

plt.tight_layout()
plt.savefig(base_path.parent / 'fig4_duration_by_topic.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(base_path.parent / 'fig4_duration_by_topic.pdf', bbox_inches='tight')
print("Figure 4 saved: Duration distribution by topic")
plt.close()

# ========== FIGURE 5: Demographics Distribution (3 PIE CHARTS - FIXED TEXT) ==========
# Extract all demographic labels (flatten lists)
all_races = []
all_genders = []
all_ages = []

for idx, row in df.iterrows():
    all_races.extend(row['race'])
    all_genders.extend(row['gender'])
    all_ages.extend(row['age'])

# Count occurrences
race_counts = Counter(all_races)
gender_counts = Counter(all_genders)
age_counts = Counter(all_ages)

# Age mapping to brackets only (remove text labels)
age_mapping = {
    'Young (18-24)': '18-24',
    'Middle (25-39)': '25-39',
    'Older (40+)': '40+',
    'Young': '18-24',
    'Middle': '25-39',
    'Older': '40+'
}

# Remap age counts
age_counts_remapped = {}
for age_label, count in age_counts.items():
    bracket = age_mapping.get(age_label, age_label)
    age_counts_remapped[bracket] = age_counts_remapped.get(bracket, 0) + count

# Create figure with 3 subplots
fig5, (ax_race, ax_gender, ax_age) = plt.subplots(1, 3, figsize=(15, 5))

# Race pie chart - FIXED TEXT POSITIONING
race_labels = list(race_counts.keys())
race_values = list(race_counts.values())
colors_race = plt.cm.Set3(np.linspace(0, 1, len(race_labels)))

wedges, texts, autotexts = ax_race.pie(race_values, autopct='%1.1f%%',
                                        colors=colors_race, startangle=90,
                                        pctdistance=0.85)
# Add legend instead of labels on the pie
ax_race.legend(wedges, race_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(9)
    autotext.set_weight('bold')
ax_race.set_title('Race', pad=10)

# Gender pie chart - FIXED TEXT POSITIONING
gender_labels = list(gender_counts.keys())
gender_values = list(gender_counts.values())
colors_gender = ['#1f77b4', '#ff7f0e']

wedges, texts, autotexts = ax_gender.pie(gender_values, labels=gender_labels, autopct='%1.1f%%',
                                          colors=colors_gender, startangle=90,
                                          labeldistance=1.15, pctdistance=0.85)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')
for text in texts:
    text.set_fontsize(10)
ax_gender.set_title('Gender', pad=10)

# Age pie chart - FIXED TEXT POSITIONING
age_labels_clean = list(age_counts_remapped.keys())
age_values_clean = list(age_counts_remapped.values())
colors_age = ['#ff7f0e', '#2ca02c', '#9467bd']

wedges, texts, autotexts = ax_age.pie(age_values_clean, labels=age_labels_clean, autopct='%1.1f%%',
                                       colors=colors_age, startangle=90,
                                       labeldistance=1.15, pctdistance=0.85)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')
for text in texts:
    text.set_fontsize(10)
ax_age.set_title('Age', pad=10)

plt.tight_layout()
plt.savefig(base_path.parent / 'fig5_demographics_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(base_path.parent / 'fig5_demographics_distribution.pdf', bbox_inches='tight')
print("Figure 5 saved: Demographics distribution")
plt.close()

# ===== Print Summary Statistics =====
print("\n" + "="*70)
print("DATASET STATISTICS SUMMARY")
print("="*70)
print(f"Total videos: {len(df)}")
print(f"Total topics: {df['topic_name'].nunique()}")
print(f"Total duration: {df['duration_seconds'].sum() / 3600:.2f} hours")

print(f"\nVideos per topic:")
for topic, count in topic_counts.items():
    hours = (df[df['topic_name'] == topic]['duration_seconds'].sum() / 3600)
    print(f"  {topic}: {count} videos ({hours:.2f} hours)")

print(f"\nDuration category distribution:")
for cat in duration_order:
    count = (df['duration_category'] == cat).sum()
    pct = (count / len(df) * 100)
    print(f"  {cat.capitalize()}: {count} ({pct:.1f}%)")

print(f"\nDemographic distribution:")
print(f"  Race: {dict(race_counts)}")
print(f"  Gender: {dict(gender_counts)}")
print(f"  Age: {dict(age_counts_remapped)}")
print("="*70)