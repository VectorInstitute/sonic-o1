# Video Question-Answer (VQA) Generation

## Overview

This directory handles automatic generation of three video QA-related tasks using Gemini-based multimodal models:
1. **Summarization** - Short + detailed summaries
2. **Multiple Choice Questions (MCQ)** - Segment-level questions with options
3. **Temporal Localization** - Segment-level time-related questions

### Directory Structure

```
04_vqa_generation/
├── main.py                      # Main VQA generation script
├── fill_empty_demographics.py   # Fill empty demographics in VQA files
├── standardize_demographics.py # Standardize demographics to canonical categories
├── vqa_config.yaml             # Configuration file
├── .env                        # API keys (create this file)
├── README.md                   # This file
│
├── models/                      # Model implementations
│   ├── base_gemini.py
│   ├── summarization_model.py
│   ├── mcq_model.py
│   ├── temporal_localization_model.py
│   └── temporal_question_judge.py
│
├── prompts/                     # Prompt templates
│   ├── summarization_prompts.py
│   ├── mcq_prompts.py
│   ├── temporal_localization_prompts.py
│   └── temporal_judge_prompts.py
│
└── utils/                       # Utility modules
    ├── demographics_expander.py
    ├── frame_sampler.py
    └── video_segmenter.py
```

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: VQA Generation                                             │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  dataset/videos/<topic>/metadata_enhanced.json                │
│         dataset/videos/<topic>/video_*.mp4                          │
│         dataset/audios/<topic>/audio_*.m4a                          │
│         dataset/captions/<topic>/caption_*.srt                      │
│ Output: vqa/                                                         │
│         ├── task1_summarization/                                    │
│         │   ├── 01_<Topic_Name>.json                                │
│         │   └── ...                                                  │
│         ├── task2_mcq/                                              │
│         │   ├── 01_<Topic_Name>.json                                │
│         │   └── ...                                                  │
│         └── task3_temporal_localization/                             │
│             ├── 01_<Topic_Name>.json                                │
│             └── ...                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Fill Empty Demographics (Optional)                          │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  vqa/task*/*.json (with empty demographics arrays)           │
│ Output: vqa/task*/*.json (with demographics filled)                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Standardize Demographics (Optional)                          │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  vqa/task*/*.json (with variant demographic terms)            │
│ Output: vqa/task*/*.json (with canonical demographic categories)    │
└─────────────────────────────────────────────────────────────────────┘
```

### Features

- **Three Task Types**: Summarization, MCQ generation, and temporal localization
- **Multimodal Analysis**: Combines video, audio, and transcript data
- **Automatic Segmentation**: Handles long videos by splitting into segments
- **Demographics Integration**: Includes demographic information in VQA entries
- **Rate Limiting**: Configurable delays to avoid API limits
- **Error Handling**: Retries, validation, and graceful degradation
- **Skip Existing**: Automatically skips videos that already have VQA generated

## Prerequisites

Before running this step, you must have completed:

1. **Data Curation** (see [01_data_curation](../01_data_curation/))
   - Downloaded videos and audio files
   - Generated metadata.json files

2. **Caption Generation** (see [02_caption_generation](../02_caption_generation/))
   - Generated captions for all videos (SRT format)

3. **Demographics Annotation** (see [03_demographics_annotation](../03_demographics_annotation/))
   - Generated metadata_enhanced.json files with demographics

Your `dataset/` directory should have this structure:
```
dataset/
├── videos/<topic_name>/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── metadata_enhanced.json
├── audios/<topic_name>/
│   ├── audio_001.m4a
│   └── audio_002.m4a
└── captions/<topic_name>/
    ├── caption_001.srt
    └── caption_002.srt
```

4. **API Setup**
   1. **Get Gemini API Key** (Required)
      - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
      - Create or select a project
      - Generate an API key

   2. **Get OpenAI API Key** (Optional)
      - Required only if temporal localization uses OpenAI-based judging/validation
      - Go to [OpenAI Platform](https://platform.openai.com/api-keys)
      - Create an API key

   3. **Set API Keys**
      Create a `.env` file in this directory:
      ```bash
      GEMINI_API_KEY=your_gemini_api_key_here
      OPENAI_API_KEY=your_openai_api_key_here  # Optional
      ```

      Or export them as environment variables:
      ```bash
      export GEMINI_API_KEY=your_gemini_api_key_here
      export OPENAI_API_KEY=your_openai_api_key_here  # Optional
      ```

## Installation

### Required Packages

All required Python packages are included in the project's
[requirements_venv.txt](../../requirements_venv.txt).


## Configuration

Edit [vqa_config.yaml](vqa_config.yaml) to customize processing settings.

### Model Settings
```yaml
gemini:
  model_name: "gemini-2.5-flash"  # Model to use
  temperature: 0.3                # Lower = more deterministic
  max_output_tokens: 2048         # Response length
  retry_attempts: 3               # Number of retries on failure
  file_processing_timeout: 7200   # Max time for file processing (2 hours)
```

### Video Processing Settings
```yaml
video:
  summarization_segment_duration: 600      # 10 minutes for summarization
  mcq_segment_duration: 180               # 3 minutes for MCQ
  temporal_localization_segment_duration: 180  # 3 minutes for temporal
  segment_overlap: 30                     # Overlap between segments (30 sec)
```

### Rate Limiting
```yaml
rate_limit:
  delay_between_videos: 45        # Seconds between videos
  delay_after_segment: 10         # Seconds after each segment
  delay_after_long_video: 90      # Extra delay after long videos
  long_video_threshold: 1800      # Threshold for "long" video (30 min)
  delay_after_api_call: 15        # Seconds after each Gemini call
```

### Task-Specific Settings
```yaml
summarization:
  constraints:
    max_words_detailed: 300
    max_words_segment: 120
    timeline_items_min: 5
    timeline_items_max: 12

mcq:
  num_options: 5                  # Always 5 options
  questions_per_segment: 1         # MCQs per segment

temporal_localization:
  questions_per_segment: 3         # Temporal questions per segment
  judge_enabled: true              # Enable GPT-4V validation (optional)
  judge_model: "gpt-4o"            # OpenAI model for validation
```

### Processing Options
```yaml
processing:
  save_raw_responses: true         # Save raw Gemini responses for debugging
  skip_existing: true              # Skip videos that already have VQA generated
  parallel_processing: false      # Set to true for faster processing
```

## Usage

**IMPORTANT**: Always run the scripts from the project root
(sonic-o1/sonic-o1 directory) so relative paths work correctly.

### Process All Topics - All Tasks

```bash
# Navigate to working directory (note: sonic-o1/sonic-o1)
cd /path/to/sonic-o1/sonic-o1

# Run VQA generation for all topics
python 04_vqa_generation/main.py --all
```

### Process Specific Topics

```bash
# Process topics 1, 2, and 3
python 04_vqa_generation/main.py --topics 1,2,3

# Process single topic
python 04_vqa_generation/main.py --topics 5
```

### Process Specific Task Only

```bash
# Generate only summarization for topics 1 and 2
python 04_vqa_generation/main.py --topics 1,2 --task summarization

# Generate only MCQ for all topics
python 04_vqa_generation/main.py --all --task mcq

# Generate only temporal localization for all topics
python 04_vqa_generation/main.py --all --task temporal
```

### Use Custom Configuration

```bash
python 04_vqa_generation/main.py --config path/to/custom_config.yaml
```

### Fill Empty Demographics

After generating VQA, fill empty demographics arrays:

```bash
# Dry run to see what would be filled
python 04_vqa_generation/fill_empty_demographics.py --dry-run

# Fill demographics for all topics
python 04_vqa_generation/fill_empty_demographics.py

# Fill demographics for specific topics
python 04_vqa_generation/fill_empty_demographics.py --topics 10,11
```

### Standardize Demographics

Standardize demographic values to canonical categories:

```bash
# Dry run to see what would be standardized
python 04_vqa_generation/standardize_demographics.py --dry-run

# Standardize demographics for all topics
python 04_vqa_generation/standardize_demographics.py

# Standardize demographics for specific topics
python 04_vqa_generation/standardize_demographics.py --topics 1,2,3
```

### Command-Line Arguments

The main script supports several command-line arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Path to configuration file (default: `vqa_config.yaml`) | `--config my_config.yaml` |
| `--topics` | Comma-separated topic IDs to process | `--topics 1,2,3` |
| `--all` | Process all topics | `--all` |
| `--task` | Process specific task only | `--task summarization` |
| `--output` | Output directory (overrides config) | `--output custom_output/` |

**Examples:**

```bash
# Process single topic with custom config
python 04_vqa_generation/main.py \
  --topics 1 \
  --config custom_config.yaml

# Process specific task for multiple topics
python 04_vqa_generation/main.py \
  --topics 1,2,3 \
  --task mcq \
  --output vqa_custom/
```

## Output

The script creates JSON files in the configured output directory (default: `vqa/`)
organized by task type, with one JSON file per topic.

### Output Location
```
vqa/
├── task1_summarization/
│   ├── 01_Patient-Doctor_Consultations.json
│   ├── 02_Job_Interviews.json
│   └── ...
├── task2_mcq/
│   ├── 01_Patient-Doctor_Consultations.json
│   ├── 02_Job_Interviews.json
│   └── ...
└── task3_temporal_localization/
    ├── 01_Patient-Doctor_Consultations.json
    ├── 02_Job_Interviews.json
    └── ...
```

### Output Format

Each output file has a shared wrapper structure:

```json
{
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "task": "summarization",
  "generated_at": "2026-01-14 12:34:56",
  "num_entries": 25,
  "entries": []
}
```

### Task 1: Summarization (`task1_summarization/*.json`)

`entries` is a list with one entry per video:

```json
{
  "video_id": "001",
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "summary_short": [
    "Bullet point 1",
    "Bullet point 2",
    "..."
  ],
  "summary_detailed": "Full detailed summary text...",
  "timeline": [
    {"time": "00:05", "event": "Doctor introduces himself"},
    {"time": "00:12", "event": "Patient describes symptoms"}
  ],
  "glossary": [
    {"term": "Hypertension", "definition": "High blood pressure"},
    {"term": "Systolic", "definition": "Upper blood pressure reading"}
  ],
  "demographics": {
    "race": ["White"],
    "gender": ["Male", "Female"],
    "age": ["Middle (25-39)"],
    "language": ["English"]
  },
  "confidence": 0.92
}
```

### Task 2: MCQ (`task2_mcq/*.json`)

`entries` is a list with one entry per generated question (typically multiple per video):

```json
{
  "video_id": "001",
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "segment": {
    "start": 120.0,
    "end": 180.0
  },
  "question": "What is the patient's primary concern?",
  "options": [
    "High blood pressure",
    "Chest pain",
    "Headache",
    "Fatigue",
    "Not enough evidence"
  ],
  "correct_answer": 0,
  "evidence_tags": ["medical_monitors", "beds"],
  "demographics": {
    "race": ["White"],
    "gender": ["Male"],
    "age": ["Middle (25-39)"],
    "language": ["English"]
  },
  "confidence": 0.85
}
```

### Task 3: Temporal Localization (`task3_temporal_localization/*.json`)

`entries` is a list with one entry per generated temporal question:

```json
{
  "video_id": "001",
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "segment": {
    "start": 45.0,
    "end": 90.0
  },
  "question": "What happens after the doctor takes the patient's blood pressure?",
  "answer": "The doctor reviews the readings and discusses treatment options.",
  "temporal_relation": "after",
  "demographics": {
    "race": ["White"],
    "gender": ["Male", "Female"],
    "age": ["Middle (25-39)"],
    "language": ["English"]
  },
  "confidence": 0.81
}
```

## Processing Time

- **Per video**: ~30-120 seconds depending on video length and task type
- **Long videos**: Automatically segmented and may take longer
- **Rate limiting**: Script includes delays between videos to avoid API limits

### Estimated Time for Full Dataset
- 13 topics × 25 videos = 325 videos
- Average 60 seconds per video = ~325 minutes
- With rate limiting: ~6-8 hours total (all tasks)

## Troubleshooting

### API Key Not Found

**Problem**: `ERROR: API key not set!`

**Solution**:
```bash
# Create .env file in 04_vqa_generation directory
echo "GEMINI_API_KEY=your_key_here" > 04_vqa_generation/.env

# Or export environment variable
export GEMINI_API_KEY=your_key_here
```

### File Not Found Errors

**Problem**: `FileNotFoundError: dataset/videos/...`

**Solution**: Make sure you're running from the project root:
```bash
cd /path/to/sonic-o1/sonic-o1
python 04_vqa_generation/main.py --all
```

### API Rate Limit Exceeded

**Problem**: `429 Too Many Requests`

**Solution**: Increase delays in [vqa_config.yaml](vqa_config.yaml):
```yaml
rate_limit:
  delay_between_videos: 60      # Increase from 45
  delay_after_api_call: 20     # Increase from 15
  delay_after_long_video: 120   # Increase from 90
```

### Empty Demographics

**Problem**: Some VQA entries have empty demographics arrays

**Solution**: Run the fill demographics script:
```bash
python 04_vqa_generation/fill_empty_demographics.py
```

### Video Too Long

**Problem**: Processing takes too long or times out

**Solution**: Adjust segment duration in [vqa_config.yaml](vqa_config.yaml):
```yaml
video:
  summarization_segment_duration: 300  # Reduce from 600
  mcq_segment_duration: 120            # Reduce from 180
```

### Timeout Errors

**Problem**: `TimeoutError: Processing timed out`

**Solution**: Increase timeout in [vqa_config.yaml](vqa_config.yaml):
```yaml
gemini:
  file_processing_timeout: 10800  # Increase from 7200 (3 hours)
```

## Files

- `main.py` - Main VQA generation script
- `fill_empty_demographics.py` - Fill empty demographics in VQA files
- `standardize_demographics.py` - Standardize demographics to canonical categories
- `vqa_config.yaml` - Configuration file
- `models/` - Model implementations (base_gemini, summarization, mcq, temporal)
- `prompts/` - Prompt templates for each task
- `utils/` - Utility modules (demographics_expander, frame_sampler, video_segmenter)
- `.env` - Environment variables (API keys) - create this file

## Notes

- The script processes videos in order by video number
- Already processed videos are skipped automatically if `skip_existing: true`
- Raw model responses are saved for debugging if `save_raw_responses: true`
- Temporal localization uses GPT-4V for validation (optional, requires OpenAI key)
- All paths are relative to the project root, so always run from the
  sonic-o1/sonic-o1 directory (the inner sonic-o1 directory that
  contains the pipeline code)
- Demographics optimization: Task 2 reuses demographics from Task 3
  (same segments), Task 1 generates separately for full videos
