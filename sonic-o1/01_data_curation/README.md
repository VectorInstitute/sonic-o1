# Data Curation Pipeline

This directory contains the YouTube video metadata scraping and parsing pipeline for the sonic-o1 dataset.

## Overview

The data curation process consists of two main steps:
1. **YouTube Metadata Scraping** - Collect video metadata from YouTube based on topics and demographics
2. **Topic Parsing** - Process and filter the metadata to create quality-annotated datasets

## Prerequisites

### Required Packages
All required Python packages are already included in the project's [requirements_venv.txt](../../requirements_venv.txt):
- `google-api-python-client` - YouTube Data API
- `youtube-transcript-api` - Caption/transcript download
- `pyyaml` - Configuration file parsing
- `python-dotenv` - Environment variable management
- `pandas` - Data processing
- `yt-dlp` - Video downloading

### Additional Requirements
- **ffmpeg** - Required for audio extraction (install separately)
  ```bash
  # Linux
  sudo apt-get install ffmpeg

  # macOS
  brew install ffmpeg

  # Conda
  conda install -c conda-forge ffmpeg
  ```

### API Setup
1. Get a YouTube Data API v3 key from [Google Cloud Console](https://console.cloud.google.com/)
2. Create a `.env` file in this directory:
```bash
YT_SCRAP_API3=your_youtube_api_key_here
```

### Configuration
Edit [config.yaml](config.yaml) to customize:
- API rate limits and search parameters
- Directory paths
- Video filtering criteria (duration, quality, demographics)
- Collection targets (videos per topic/query)

## Step 1: YouTube Metadata Scraping

The [youtube_metadata_scraper.py](youtube_metadata_scraper.py) script collects video metadata from YouTube across 13 topics with demographic diversity.

### Features
- Searches for videos across multiple topics (Patient-Doctor Consultations, Job Interviews, etc.)
- Generates demographically diverse queries (race, gender, age, language)
- Filters for quality using engagement metrics and clickbait detection
- Collects comprehensive metadata (views, likes, captions, duration, etc.)
- Supports incremental collection (adds new videos without duplicates)

### Usage

```bash
python youtube_metadata_scraper.py
```

### Configuration Parameters

Key settings in `config.yaml`:
- `videos_per_topic`: Target number of videos per topic (default: 100)
- `videos_per_query`: Videos to retrieve per search query (default: 15)
- `video_duration`: Filter by duration - "short", "medium", "long", or "any"
- `years_back`: How many years back to search (default: 5)
- `video_license`: License filter - "creativeCommon" or "any"

### Output

The scraper creates a directory structure:
```
videos_Unfiltered/
├── 01_Patient-Doctor_Consultations/
│   ├── Patient-Doctor_Consultations_metadata.json
│   ├── Patient-Doctor_Consultations_metadata.csv
│   └── Patient-Doctor_Consultations_summary.json
├── 02_Job_Interviews/
│   └── ...
└── all_topics_combined.csv
```

### Batch Processing

The main function processes topics in ranges. Edit lines 995-1000 to process different topic batches:
```python
for topic_id in range(1, 4):  # Day 1: Topics 1-3
    # Change to (4, 7), (7, 10), etc. for subsequent runs
```

## Step 2: Topic Parsing

**IMPORTANT PREREQUISITE**: Before running [parse_topic.py](parse_topic.py), you MUST:

1. **Create the `videos_QualityAnnotated` directory** following the structure from [huggingface_review_template](../huggingface_review_template/)
2. **Manually review and quality-annotate** the metadata files from Step 1
3. **Add quality labels** to each video in the metadata JSON files
   - The parse script filters videos based on `Qualitylabel: "Good"`
   - Review videos to ensure they fit the topic and meet quality standards

### What parse_topic.py Does

The [parse_topic.py](parse_topic.py) script:
- Loads quality-annotated metadata from `videos_QualityAnnotated/`
- Filters videos based on:
  - Quality label (`Qualitylabel == "Good"`)
  - Copyright notice (`copyright_notice == "creativeCommon"`)
  - Language (English audio or default language)
- Downloads videos, extracts audio, and downloads captions
- Creates diverse selections across demographics and durations
- Manages incremental additions (max 25 videos per topic by default)

### Directory Structure Required

Before running parse_topic.py, ensure this structure exists:
```
videos_QualityAnnotated/
├── 01_Patient-Doctor_Consultations/
│   └── Patient-Doctor_Consultations_metadata.json  # With Qualitylabel field added
├── 02_Job_Interviews/
│   └── Job_Interviews_metadata.json
└── ...
```

### Usage

```bash
python parse_topic.py
```

### Configuration

Edit the `MAX_COUNT` variable in parse_topic.py (line 541) to set maximum videos per topic:
```python
MAX_COUNT = 25  # Change to your desired max videos per topic
```

### Output Structure

The script creates a `dataset/` directory:
```
dataset/
├── videos/
│   └── 01_Patient-Doctor_Consultations/
│       ├── video_001.mp4
│       ├── video_002.mp4
│       └── metadata.json
├── audios/
│   └── 01_Patient-Doctor_Consultations/
│       ├── audio_001.m4a
│       ├── audio_002.m4a
│       └── metadata.json
└── captions/
    └── 01_Patient-Doctor_Consultations/
        ├── caption_001.srt
        ├── caption_002.srt
        ├── needs_whisper.txt
        └── metadata.json
```

### Missing Captions

Videos without available captions are tracked in `needs_whisper.txt` files within each topic's caption directory. These will need transcription in the next processing step (see `02_` directory for transcription workflow).

## Complete Workflow

1. **Setup**
   ```bash
   # Install ffmpeg (if not already installed)
   # See "Additional Requirements" section above for your platform

   # Create .env file with your YouTube API key
   echo "YT_SCRAP_API3=your_api_key" > .env
   ```

2. **Configure** - Edit [config.yaml](config.yaml) with your preferences

3. **Scrape Metadata**
   ```bash
   python youtube_metadata_scraper.py
   # Output: videos_Unfiltered/ with metadata files
   ```

4. **Quality Review** (Manual Step)
   - Review videos using the tools in [huggingface_review_template](../huggingface_review_template/)
   - Add `Qualitylabel` field to metadata
   - Create `videos_QualityAnnotated/` directory with annotated metadata

5. **Parse and Download**
   ```bash
   python parse_topic.py
   # Output: dataset/ with videos, audios, captions
   ```

6. **Next Steps**
   - Videos needing transcription are listed in `dataset/captions/*/needs_whisper.txt`
   - Proceed to the next pipeline stage for transcription (see `02_` directory)

## Topics Covered

1. Patient-Doctor Consultations
2. Job Interviews
3. Parent-Teacher Conferences
4. Customer Service Interactions
5. Courtroom Proceedings
6. Emergency Response Scenarios
7. Public Transportation Conflicts
8. Workplace Team Meetings
9. Housing/Apartment Tours
10. Restaurant Service Encounters
11. Mental Health Counseling
12. Community Town Halls
13. Olympics

## Quality Filtering

The scraper applies research-based quality filters:
- **Duration**: 30 seconds to 60 minutes (configurable)
- **Engagement**: Minimum views, like/comment ratios
- **Clickbait Detection**: Filters extreme clickbait patterns
- **Spam Detection**: Removes spam and low-quality content
- **License**: Filters by Creative Commons license (default)

## Diversity Considerations

The pipeline ensures demographic diversity through:
- Multi-dimensional search queries (race, gender, age, language)
- Balanced selection across demographics
- Duration category distribution (40% short, 40% medium, 20% long)

## Troubleshooting

### "No videos meet the criteria"
- Check that videos in `videos_QualityAnnotated/` have `Qualitylabel: "Good"`
- Verify copyright_notice is set to "creativeCommon"
- Check language fields (default_language or default_audio_language should be "en")

### API Rate Limiting
- Increase `rate_limit_delay` in config.yaml
- Process topics in smaller batches

### ffmpeg not found
See "Additional Requirements" section above for installation instructions.

## File Descriptions

- [youtube_metadata_scraper.py](youtube_metadata_scraper.py) - Main scraper for YouTube metadata collection
- [parse_topic.py](parse_topic.py) - Video download and processing script
- [config.yaml](config.yaml) - Configuration file for both scripts
- [.env](.env) - Environment variables (API keys)

## Notes

- The scraper supports incremental collection - running it multiple times will add new videos without duplicates
- Quality filtering helps ensure dataset integrity for research purposes
- Always respect YouTube's Terms of Service and copyright laws
- Consider computational resources when setting `MAX_COUNT` - video processing is storage and compute intensive
