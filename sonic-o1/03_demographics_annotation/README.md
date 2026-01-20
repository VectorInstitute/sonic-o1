# Demographics Annotation with Gemini

This directory handles automatic demographics annotation for videos using Google's Gemini multimodal model. It analyzes videos, audio, and captions to extract demographic information (race, gender, age, language) of people appearing in the videos.

## Prerequisites

Before running this step, you must have completed:
1. **Data Curation** (see [01_data_curation](../01_data_curation/)) - Downloaded videos and audio
2. **Caption Generation** (see [02_caption_generation](../02_caption_generation/)) - Generated captions for all videos

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

## Required Packages

All required Python packages are already included in the project's [requirements_venv.txt](../../requirements_venv.txt):
- `google-generativeai` - Gemini API
- `python-dotenv` - Environment variable management
- `pyyaml` - Configuration file parsing
- `tqdm` - Progress bars

## API Setup

### Get Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or select a project
3. Generate an API key

### Set API Key
Create a `.env` file in this directory:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Or export it as an environment variable:
```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

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
  base_path: "dataset"      # Path to dataset directory
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
```

### Rate Limiting
```yaml
rate_limit:
  delay_between_videos: 15      # Seconds between videos
  delay_after_long_video: 60    # Extra delay after long videos
  long_video_threshold: 1800    # Threshold for "long" video (30 min)
```

## Usage

**IMPORTANT**: Always run the annotation script from the project root (sonic-o1/sonic-o1 directory) so relative paths work correctly.

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

### Test Single Video

To test on a single video before processing everything:

```bash
# Edit test_single_video.py to set topic and video number
# Lines 25-26:
#   topic = "01_Patient-Doctor_Consultations"
#   video_number = "015"

# Run test from project root
python 03_demographics_annotation/test_single_video.py
```

### Use Custom Configuration

```bash
python 03_demographics_annotation/run_annotation.py --config path/to/custom_config.yaml
```

## Output

The script creates `metadata_enhanced.json` files in each topic directory with demographic annotations:

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
  "raw_response": "...",
  "processing_timestamp": "2024-01-14T12:00:00"
}
```

### Output Location
```
dataset/
└── videos/<topic_name>/
    ├── metadata.json                           # Original metadata
    ├── metadata_enhanced.json                  # With demographics annotations
    └── metadata_enhanced_checkpoint.json       # Checkpoint for resume
```

## Checkpoint and Resume

The script automatically saves checkpoints every N videos (configured by `save_interval`). If the script is interrupted:

1. **Resume automatically** - Just run the script again, it will detect the checkpoint and continue where it left off
2. **Start fresh** - Delete the `metadata_enhanced_checkpoint.json` file in the topic directory

```bash
# To start fresh on a specific topic
rm dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced_checkpoint.json
```

## Examples

### Example 1: Process All Topics

```bash
# 1. Navigate to working directory
cd /path/to/sonic-o1/sonic-o1

# 2. Set API key (if not in .env)
export GEMINI_API_KEY=your_key_here

# 3. Run annotation
python 03_demographics_annotation/run_annotation.py
```

### Example 2: Process Only New Topics

Edit [config.yaml](config.yaml):
```yaml
dataset:
  topics:
    - "11_Mental_Health_Counseling"
    - "12_Community_Town_Halls"
```

Run:
```bash
python 03_demographics_annotation/run_annotation.py
```

### Example 3: Test Single Video First

```bash
# Edit test_single_video.py to set your test video
# topic = "02_Job_Interviews"
# video_number = "005"

# Run test
python 03_demographics_annotation/test_single_video.py
```

Expected output:
```
================================================================================
MULTIMODAL DEMOGRAPHICS ANNOTATION TEST
================================================================================
Topic: 02_Job_Interviews
Item Number: 005
Model: gemini-2.5-flash
================================================================================

File Paths:
  Video:    dataset/videos/02_Job_Interviews/video_005.mp4
  Audio:    dataset/audios/02_Job_Interviews/audio_005.m4a
  Caption:  dataset/captions/02_Job_Interviews/caption_005.srt

Processing...
[Success] Demographics extracted
```

### Example 4: Resume After Interruption

```bash
# If script was interrupted, just run again
python 03_demographics_annotation/run_annotation.py

# Output will show:
# "Found checkpoint file: metadata_enhanced_checkpoint.json"
# "Loaded 15 processed videos from checkpoint"
# "Resuming from video 16/25"
```

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

**Solution**: The script automatically retries failed videos. Check the log file:
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

## Quality Control

The script includes built-in quality checks:

1. **Validation**: Ensures demographics match allowed categories
2. **Retry Logic**: Automatically retries failed videos
3. **Checkpointing**: Saves progress to prevent data loss
4. **Logging**: Detailed logs for debugging

### Check Annotation Quality

```bash
# Count videos with demographics
jq '[.[] | select(.demographics_detailed != null)] | length' \
  dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced.json

# View specific annotation
jq '.[] | select(.video_number == "001")' \
  dataset/videos/01_Patient-Doctor_Consultations/metadata_enhanced.json
```

## File Descriptions

- [run_annotation.py](run_annotation.py) - Main annotation pipeline script
- [test_single_video.py](test_single_video.py) - Test script for single video
- [config.yaml](config.yaml) - Configuration file
- [config_loader.py](config_loader.py) - Configuration loader
- [model.py](model.py) - Gemini API wrapper and demographics extraction
- [prompts.py](prompts.py) - System and user prompts for the model
- [.env](.env) - Environment variables (API keys)

## Notes

- The script processes videos in order by video number
- Already processed videos are skipped automatically
- Raw model responses are saved for debugging if `save_raw_responses: true`
- Backups are created before overwriting if `create_backup: true`
- All paths are relative to the project root, so always run from the sonic-o1/sonic-o1 directory (the inner sonic-o1 directory that contains the pipeline code)
