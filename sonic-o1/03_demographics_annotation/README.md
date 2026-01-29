# Demographics Annotation with Gemini

## Overview

This directory handles automatic demographics annotation for videos using Google's Gemini multimodal model. It analyzes videos, audio, and captions to extract demographic information (race, gender, age, language) of people appearing in the videos.

### Directory Structure

```
03_demographics_annotation/
├── run_annotation.py      # Main annotation pipeline script
├── config_loader.py        # Configuration loader (YAML + .env)
├── config.yaml            # Configuration file
├── model.py               # Gemini API wrapper
├── prompts.py             # Prompt templates
├── README.md              # This file
│
└── dataset/              # Output directory (from 01_data_curation)
    └── videos/<topic_name>/
        ├── video_001.mp4
        ├── metadata.json
        └── metadata_enhanced.json  # Generated output
```

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Demographics Annotation                                     │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  dataset/videos/<topic>/                                     │
│         dataset/audios/<topic>/                                     │
│         dataset/captions/<topic>/                                   │
│ Output: dataset/videos/<topic>/metadata_enhanced.json               │
│         ├── demographics_detailed: {race, gender, age, language}    │
│         ├── demographics_confidence: confidence scores              │
│         └── demographics_annotation: metadata                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Next Steps                                                  │
│ ──────────────────────────────────────────────────────────────────  │
│ • Check metadata_enhanced.json for demographic annotations          │
│ • Use --retry-failed to reprocess videos with empty demographics    │
│ • Proceed to 04_* directory for VQA generation                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Features

- **Multimodal Analysis**: Combines video, audio, and transcript data
- **Automatic Segmentation**: Handles long videos (>25 min) by splitting
- **Checkpoint/Resume**: Saves progress every N videos
- **Retry Failed**: Reprocess videos with empty demographics
- **Rate Limiting**: Configurable delays to avoid API limits
- **Error Handling**: Retries, validation, and graceful degradation

### Quality Control

The pipeline includes built-in quality assurance mechanisms:

- **Validation**: Ensures demographics match allowed categories from config
- **Confidence Filtering**: Filters low-confidence annotations based on `min_confidence` threshold
- **Retry Logic**: Automatically retries failed API calls (up to `retry_attempts`)
- **Checkpointing**: Saves progress periodically to prevent data loss on interruption
- **Comprehensive Logging**: Detailed logs for debugging and quality monitoring
- **Error Recovery**: Graceful handling of API errors, timeouts, and invalid responses

## Prerequisites

Before running this step, you must have completed:

1. **Data Curation** (see [01_data_curation](../01_data_curation/))
   - Downloaded videos and audio files
   - Generated metadata.json files

2. **Caption Generation** (see [02_caption_generation](../02_caption_generation/))
   - Generated captions for all videos (SRT format)

Your `dataset/` directory should have this structure:
```
dataset/
├── videos/<topic_name>/
│   ├── video_001.mp4
│   └── metadata.json
├── audios/<topic_name>/
│   └── audio_001.m4a
└── captions/<topic_name>/
    └── caption_001.srt
```

3. **API Setup**
   1. **Get Gemini API Key**
      - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
      - Create or select a project
      - Generate an API key

   2. **Set API Key**
      Create a `.env` file in this directory:
      ```bash
      GEMINI_API_KEY=your_gemini_api_key_here
      ```

      Or export it as an environment variable:
      ```bash
      export GEMINI_API_KEY=your_gemini_api_key_here
      ```

## Installation

### Required Packages

All required Python packages are already included in the project's
[requirements_venv.txt](../../requirements_venv.txt):

## Configuration

Edit [config.yaml](config.yaml) to customize:

### Model Settings
```yaml
model:
  name: "gemini-2.5-flash"  # Model to use
  temperature: 0.3          # Lower = more deterministic
  max_output_tokens: 1024   # Response length
  timeout: 60               # API timeout in seconds
  retry_attempts: 3         # Number of retries on failure
```

### Dataset Settings
```yaml
dataset:
  base_path: "../dataset"   # Path to dataset directory
  topics:                   # Topics to process (or leave empty for all)
    - "01_Patient-Doctor_Consultations"
    - "02_Job_Interviews"
```

### Processing Settings
```yaml
processing:
  batch_size: 5             # Videos to process before saving
  save_interval: 10         # Save checkpoint every N videos
  max_video_duration: 1500  # Max duration before segmentation (25 min)
  enable_segmentation: true # Auto-segment long videos
  prefer_video_with_audio: false  # Send both video AND audio

  # Output settings
  save_raw_responses: true  # Save raw API responses for debugging
  create_backup: true       # Create backup before overwriting metadata
```

### Rate Limiting
```yaml
rate_limit:
  delay_between_videos: 15      # Seconds between videos
  delay_after_long_video: 60    # Extra delay after long videos
  long_video_threshold: 1800     # Threshold for "long" video (30 min)
```

## Usage

**IMPORTANT**: Always run the annotation script from the project root
(sonic-o1/sonic-o1 directory) so relative paths work correctly.

### Process All Topics

```bash
# Navigate to working directory (note: sonic-o1/sonic-o1)
cd /path/to/sonic-o1/sonic-o1

# Run annotation
python 03_demographics_annotation/run_annotation.py
```

### Process Specific Topics

Edit [config.yaml](config.yaml) to specify which topics to process:
```yaml
dataset:
  topics:
    - "01_Patient-Doctor_Consultations"
    - "02_Job_Interviews"
```

Then run:
```bash
python 03_demographics_annotation/run_annotation.py
```

### Process Single Topic

```bash
python 03_demographics_annotation/run_annotation.py --topic "01_Patient-Doctor_Consultations"
```

### Retry Failed Videos

Reprocess videos with empty demographics:
```bash
python 03_demographics_annotation/run_annotation.py --retry-failed
```

### Use Custom Configuration

```bash
python 03_demographics_annotation/run_annotation.py --config path/to/custom_config.yaml
```

### Command-Line Arguments

The script supports several command-line arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Path to configuration file (default: `config.yaml`) | `--config my_config.yaml` |
| `--topic` | Process specific topic only | `--topic "01_Patient-Doctor_Consultations"` |
| `--api-key` | Override Gemini API key from config/env | `--api-key "your_key_here"` |
| `--no-cache` | Reprocess all videos even if already done | `--no-cache` |
| `--retry-failed` | Only reprocess videos with empty demographics | `--retry-failed` |

**Examples:**

```bash
# Process single topic with custom config
python 03_demographics_annotation/run_annotation.py \
  --topic "01_Patient-Doctor_Consultations" \
  --config custom_config.yaml

# Retry failed videos with API key override
python 03_demographics_annotation/run_annotation.py \
  --retry-failed \
  --api-key "new_api_key"

# Reprocess all videos (ignore checkpoints)
python 03_demographics_annotation/run_annotation.py --no-cache
```

## Output

The script creates `metadata_enhanced.json` files in each topic directory
with demographic annotations:

```json
{
  "video_id": "abc123",
  "video_number": "001",
  "demographics_detailed": {
    "race": ["Asian", "White"],
    "gender": ["Male", "Female"],
    "age": ["Middle (25-39)"],
    "language": ["English"]
  },
  "demographics_confidence": {
    "race": {"Asian": 0.9, "White": 0.85},
    "gender": {"Male": 0.95, "Female": 0.9},
    "age": {"Middle (25-39)": 0.8},
    "language": {"English": 1.0}
  },
  "demographics_annotation": {
    "model": "gemini-2.5-flash",
    "annotated_at": "2024-01-14 12:00:00",
    "individuals_count": 2,
    "modalities_used": ["video", "audio", "transcript"],
    "explanation": "Video shows 2 individuals having a conversation..."
  }
}
```

### Output Location
```
dataset/
└── videos/<topic_name>/
    ├── metadata.json                           # Original metadata
    ├── metadata_enhanced.json                  # With demographics
    ├── metadata_enhanced_checkpoint.json       # Checkpoint (auto-deleted)
    └── raw_responses/                          # Raw API responses (optional)
        └── video_001_response.json
```

## Checkpoint and Resume

The script automatically saves checkpoints every N videos (configured by
`save_interval`). If the script is interrupted:

1. **Resume automatically** - Just run the script again, it will detect
   the checkpoint and continue where it left off

2. **Start fresh** - To restart processing from the beginning:
   ```bash
   # Delete checkpoint file for a specific topic
   rm dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced_checkpoint.json

   # Or delete all checkpoints
   find dataset/videos -name "metadata_enhanced_checkpoint.json" -delete

   # Then run the script normally
   python 03_demographics_annotation/run_annotation.py
   ```

   **Note**: Starting fresh will reprocess all videos in the topic. If you want
   to keep existing annotations and only process new videos, don't delete the
   checkpoint - the script will automatically skip already-processed videos.



## Processing Time

- **Per video**: ~5-30 seconds depending on video length
- **Long videos (>25 min)**: Automatically segmented and may take longer
- **Rate limiting**: Script includes delays between videos to avoid API limits

### Estimated Time for Full Dataset
- 13 topics × 25 videos = 325 videos
- Average 15 seconds per video = ~81 minutes
- With rate limiting: ~2-3 hours total

## Troubleshooting

### API Key Not Found

**Problem**: `ERROR: API key not set!`

**Solution**:
```bash
# Create .env file in 03_demographics_annotation directory
echo "GEMINI_API_KEY=your_key_here" > 03_demographics_annotation/.env

# Or export environment variable
export GEMINI_API_KEY=your_key_here
```

### File Not Found Errors

**Problem**: `FileNotFoundError: dataset/videos/...`

**Solution**: Make sure you're running from the project root:
```bash
cd /path/to/sonic-o1/sonic-o1
python 03_demographics_annotation/run_annotation.py
```

### API Rate Limit Exceeded

**Problem**: `429 Too Many Requests`

**Solution**: Increase delays in [config.yaml](config.yaml):
```yaml
rate_limit:
  delay_between_videos: 30      # Increase from 15
  delay_after_long_video: 120   # Increase from 60
```

### Empty Demographics

**Problem**: Some videos have empty `demographics_detailed`

**Solution**: Use the retry flag to reprocess failed videos:
```bash
python 03_demographics_annotation/run_annotation.py --retry-failed
```

Check the log file:
```bash
cat 03_demographics_annotation/demographics_annotation.log
```

### Video Too Long

**Problem**: `Video duration exceeds maximum`

**Solution**: Enable segmentation in [config.yaml](config.yaml):
```yaml
processing:
  enable_segmentation: true
  max_video_duration: 1500      # 25 minutes
  segment_overlap: 60           # 1 minute overlap
```

### Timeout Errors

**Problem**: `TimeoutError: Processing timed out`

**Solution**: Increase timeout in [config.yaml](config.yaml):
```yaml
model:
  timeout: 120  # Increase from 60 seconds
```

### Check Annotation Quality

You can verify the quality and completeness of annotations using these commands:

**Count videos with successful annotations:**
```bash
# Count how many videos have non-empty demographics_detailed
jq '[.[] | select(.demographics_detailed != null)] | length' \
  dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced.json
```
This counts videos where `demographics_detailed` exists and is not null. A video
is considered successfully annotated if it has at least one demographic category
(race, gender, age, or language) with non-empty values.

**View a specific video's annotation:**
```bash
# View full annotation for video_001
jq '.[] | select(.video_number == "001")' \
  dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced.json
```
This displays the complete annotation for a specific video, including:
- `demographics_detailed`: Lists of detected demographics per category
- `demographics_confidence`: Confidence scores (0.0-1.0) for each detection
- `demographics_annotation`: Metadata including model used, timestamp, individual count, and explanation

**Check for videos with empty demographics:**
```bash
# Find videos that failed annotation (empty demographics)
jq '[.[] | select(.demographics_detailed.race == [] and .demographics_detailed.gender == [] and .demographics_detailed.age == [] and .demographics_detailed.language == [])] | length' \
  dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced.json
```
This identifies videos that need reprocessing with `--retry-failed`.

## Files

- `run_annotation.py` - Main annotation pipeline script
- `config.yaml` - Configuration file
- `config_loader.py` - Configuration loader with .env support
- `model.py` - Gemini API wrapper and demographics extraction
- `prompts.py` - System and user prompts for the model
- `.env` - Environment variables (API keys) - create this file

## Notes

- The script processes videos in order by video number
- Already processed videos are skipped automatically
- Raw model responses are saved for debugging if
  `save_raw_responses: true`
- Backups are created before overwriting if `create_backup: true`
- All paths are relative to the project root, so always run from the
  sonic-o1/sonic-o1 directory (the inner sonic-o1 directory that
  contains the pipeline code)
