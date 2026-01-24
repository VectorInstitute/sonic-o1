# Data Curation Pipeline

This directory contains the YouTube video metadata scraping and parsing pipeline for the **sonic-o1** dataset.

## Overview

### Directory Structure

```
01_data_curation/
├── youtube_metadata_scraper.py    # Step 1: Scrapes YouTube metadata
├── parse_topic.py                 # Step 3: Parses and downloads videos
├── config.yaml                    # Configuration file for both scripts
├── .env                           # API keys (create this file)
├── README.md                      # This file
│
├── videos_Unfiltered/             # Output from Step 1 (auto-generated)
│   ├── 01_Patient-Doctor_Consultations/
│   ├── 02_Job_Interviews/
│   └── ...
│
├── videos_QualityAnnotated/       # Output from Step 2 (manual review)
│   ├── 01_Patient-Doctor_Consultations/
│   ├── 02_Job_Interviews/
│   └── ...
│
└── dataset/                       # Output from Step 3 (auto-generated)
    ├── videos/
    ├── audios/
    └── captions/
```

### Pipeline Workflow

The data curation pipeline consists of four main steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: YouTube Metadata Scraping                                   │
│ ────────────────────────────────────────────────────────────────── │
│ Input:  config.yaml, YouTube Data API                               │
│ Output: videos_Unfiltered/                                          │
│         ├── 01_Patient-Doctor_Consultations/                        │
│         │   ├── *_metadata.json                                     │
│         │   ├── *_metadata.csv                                      │
│         │   └── *_summary.json                                      │
│         └── all_topics_combined.csv                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Manual Quality Review (Required)                            │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  videos_Unfiltered/                                          │
│ Output: videos_QualityAnnotated/                                    │
│         ├── 01_Patient-Doctor_Consultations/                        │
│         │   └── *_metadata.json (+ Qualitylabel field)              │
│         └── ...                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Topic Parsing and Download                                  │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  videos_QualityAnnotated/                                    │
│ Output: dataset/                                                    │
│         ├── videos/01_Patient-Doctor_Consultations/                 │
│         │   ├── video_001.mp4                                       │
│         │   └── metadata.json                                       │
│         ├── audios/01_Patient-Doctor_Consultations/                 │
│         │   ├── audio_001.m4a                                       │
│         │   └── metadata.json                                       │
│         └── captions/01_Patient-Doctor_Consultations/               │
│             ├── caption_001.srt                                     │
│             ├── needs_whisper.txt                                   │
│             └── metadata.json                                       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: Next Steps                                                  │
│ ──────────────────────────────────────────────────────────────────  │
│ • Check needs_whisper.txt for videos requiring transcription        │
│ • Proceed to 02_* directory for transcription workflow              │
└─────────────────────────────────────────────────────────────────────┘
```

### Topics Covered

The pipeline processes 13 diverse topics:
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

### Quality Filtering

The scraper applies research-based quality filters:
- **Duration**: 30 seconds to 60 minutes (configurable)
- **Engagement**: Minimum views, like/comment ratios
- **Clickbait Detection**: Filters extreme clickbait patterns
- **Spam Detection**: Removes spam and low-quality content
- **License**: Filters by Creative Commons license (default)

### Diversity Considerations

The pipeline ensures demographic and content diversity:
- **Multi-dimensional search queries**: race, gender, age, language
- **Balanced selection**: across demographics
- **Duration distribution**: 40% short, 40% medium, 20% long

### Features

- Collects comprehensive metadata (views, likes, captions, duration, etc.)
- Generates demographically diverse search queries
- Supports incremental collection (reruns add new videos without duplicates)
- Downloads videos, extracts audio, and downloads captions
- Creates diverse selections across demographics and durations

---

## Prerequisites

1. **Install Python packages**
   - All required Python packages are included in `../../requirements_venv.txt`

2. **Install ffmpeg** (required for audio extraction)
   ```bash
   # Linux
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Conda
   conda install -c conda-forge ffmpeg
   ```

3. **Set up YouTube Data API v3 key**
   - Get an API key from Google Cloud Console
   - Create a `.env` file in the `01_data_curation` directory:
     ```bash
     YT_SCRAP_API=your_youtube_api_key_here
     ```

4. **Configure settings**
   - Edit `config.yaml` to customize:
     - API rate limits and search parameters
     - Directory paths
     - Video filtering criteria (duration, quality, demographics)
     - Collection targets (videos per topic/query)

## Step 1: YouTube Metadata Scraping

1. **Choose topics** (optional)
   - Default: processes all 13 topics
   - To process specific topics, edit the loop in `youtube_metadata_scraper.py` (around lines 995-1000):
     ```python
     for topic_id in range(1, 13):
     ```
    - For a free tier usage a (3 topics with 25 videos per per run / Day) is optimal

2. **Configure parameters in `config.yaml`** (optional)
   - `videos_per_topic`: target videos per topic (default: 100)
   - `videos_per_query`: videos per search query (default: 15)
   - `video_duration`: "short" | "medium" | "long" | "any"
   - `years_back`: how many years back to search (default: 5)
   - `video_license`: "creativeCommon" | "any"

3. **Run the scraper**
   ```bash
   python youtube_metadata_scraper.py
   ```

4. **Output**
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

## Step 2: Manual Quality Review (Required)

1. **Create quality-annotated directory**
   - Use tools in `../huggingface_review_template/`
   - Create `videos_QualityAnnotated/` following the template structure

2. **Review and annotate videos**
   - Manually review Step 1 outputs from `videos_Unfiltered/`
   - Add a `Qualitylabel` field to each video in the per-topic metadata JSON
   - Mark high-quality videos as `Qualitylabel: "Good"`

3. **Output**
   ```
   videos_QualityAnnotated/
   ├── 01_Patient-Doctor_Consultations/
   │   └── Patient-Doctor_Consultations_metadata.json  # With Qualitylabel field
   ├── 02_Job_Interviews/
   │   └── Job_Interviews_metadata.json
   └── ...
   ```

## Step 3: Topic Parsing and Download

1. **Configure maximum videos per topic** (optional)
   - Edit `MAX_COUNT` in `parse_topic.py` (around line 541):
     ```python
     MAX_COUNT = 25
     ```

2. **Run the parser**
   ```bash
   python parse_topic.py
   ```

3. **Output**
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

## Step 4: Next Steps

1. **Check for videos needing transcription**
   - Videos missing captions are listed in `dataset/captions/*/needs_whisper.txt`

2. **Proceed to transcription workflow**
   - See `02_` directory for transcription steps

---

## Troubleshooting

### "No videos meet the criteria"
- Ensure annotated metadata has `Qualitylabel: "Good"`
- Verify `copyright_notice == "creativeCommon"`
- Check language fields (`default_language` / `default_audio_language` include `"en"`)

### API rate limiting
- Increase `rate_limit_delay` in `config.yaml`
- Process topics in smaller batches

### ffmpeg not found
- Install via the commands listed under Prerequisites section

## Files

- `youtube_metadata_scraper.py` — main YouTube metadata collection
- `parse_topic.py` — download and processing
- `config.yaml` — configuration for both scripts
- `.env` — environment variables (API keys)

## Notes

- Incremental collection is supported (reruns add new videos without duplicates)
- Quality filtering is intended to support research-grade dataset integrity
- Respect YouTube Terms of Service and copyright laws
- Video processing is storage/compute intensive; set `MAX_COUNT` accordingly
