# Data Curation Pipeline

This directory contains the YouTube video metadata scraping and parsing pipeline for the **sonic-o1** dataset.

## Overview
- The data curation process has two main steps:
  1. **YouTube Metadata Scraping** — Collect video metadata from YouTube based on topics and demographics.
  2. **Topic Parsing** — Process and filter metadata to create **quality-annotated** datasets, then download videos/audio/captions.
- **Execution note (important):** Step 2 requires a **manual quality review** step before you can run `parse_topic.py`.

## Prerequisites
1. **Required packages/environment:** All required Python packages are included in the project’s `requirements_venv.txt` (see `../../requirements_venv.txt`).
2. **Additional requirements:** `ffmpeg` (required for audio extraction; install separately)
   ```bash
   # Linux
   sudo apt-get install ffmpeg

   # macOS
   brew install ffmpeg

   # Conda
   conda install -c conda-forge ffmpeg

3. **API setup:** Get a YouTube Data API v3 key (Google Cloud Console) and create a `.env` file in this directory.

   ```bash
   # Use the environment variable name expected by your codebase.
   YT_SCRAP_API=your_youtube_api_key_here
   ```
4. **Configuration:** Edit `config.yaml` to customize:

   * API rate limits and search parameters
   * Directory paths
   * Video filtering criteria (duration, quality, demographics)
   * Collection targets (videos per topic/query)

## Execution (recommended workflow)

1. **Configure**

   * Edit `config.yaml` with your preferences.
2. **Run Step 1: scrape metadata**

   ```bash
   python youtube_metadata_scraper.py
   ```

   * Output: `videos_Unfiltered/` containing per-topic metadata files + `all_topics_combined.csv`.
3. **Manual quality review (required before Step 2)**

   * Use tools in `../huggingface_review_template/`.
   * Create `videos_QualityAnnotated/` following the template structure.
   * Add a `Qualitylabel` field to each video in the per-topic metadata JSON.
   * The parsing step filters for `Qualitylabel == "Good"`.
4. **Run Step 2: parse + download**

   ```bash
   python parse_topic.py
   ```

   * Output: `dataset/` containing `videos/`, `audios/`, and `captions/`.
5. **Next steps**

   * Videos missing captions are listed in `dataset/captions/*/needs_whisper.txt`.
   * Proceed to the transcription workflow (see `02_` directory).

## Step 1 — YouTube Metadata Scraping (`youtube_metadata_scraper.py`)

* Purpose: Collect video metadata from YouTube across 13 topics with demographic diversity.
* Features:

  * Searches across multiple topics (Patient-Doctor Consultations, Job Interviews, etc.)
  * Generates demographically diverse queries (race, gender, age, language)
  * Filters for quality using engagement metrics and clickbait detection
  * Collects comprehensive metadata (views, likes, captions, duration, etc.)
  * Supports incremental collection (adds new videos without duplicates)
* Run:

  ```bash
  python youtube_metadata_scraper.py
  ```
* Key `config.yaml` settings:

  * `videos_per_topic`: target videos per topic (default: 100)
  * `videos_per_query`: videos per search query (default: 15)
  * `video_duration`: "short" | "medium" | "long" | "any"
  * `years_back`: how many years back to search (default: 5)
  * `video_license`: "creativeCommon" | "any"
* Output structure:

  ```text
  videos_Unfiltered/
  ├── 01_Patient-Doctor_Consultations/
  │   ├── Patient-Doctor_Consultations_metadata.json
  │   ├── Patient-Doctor_Consultations_metadata.csv
  │   └── Patient-Doctor_Consultations_summary.json
  ├── 02_Job_Interviews/
  │   └── ...
  └── all_topics_combined.csv
  ```
* Batch processing:

  * The main function processes topics in ranges (edit the loop near the bottom of the script, e.g. lines ~995–1000):

    ```python
    for topic_id in range(1, 13):  
    ```

## Step 2 — Topic Parsing (`parse_topic.py`)

* **Prerequisite reminder:** before running, you must:

  * Create `videos_QualityAnnotated/` following `../huggingface_review_template/`
  * Manually review/annotate Step 1 outputs
  * Add quality labels in the JSON (`Qualitylabel: "Good"` for selected videos)
* What it does:

  * Loads quality-annotated metadata from `videos_QualityAnnotated/`
  * Filters videos by:

    * `Qualitylabel == "Good"`
    * `copyright_notice == "creativeCommon"`
    * Language: English audio or default language
  * Downloads videos, extracts audio, downloads captions
  * Creates diverse selections across demographics and durations
  * Supports incremental additions (default max 25 videos/topic)
* Required input structure:

  ```text
  videos_QualityAnnotated/
  ├── 01_Patient-Doctor_Consultations/
  │   └── Patient-Doctor_Consultations_metadata.json  # With Qualitylabel field added
  ├── 02_Job_Interviews/
  │   └── Job_Interviews_metadata.json
  └── ...
  ```
* Run:

  ```bash
  python parse_topic.py
  ```
* Max videos/topic:

  * Edit `MAX_COUNT` in `parse_topic.py` (around line ~541):

    ```python
    MAX_COUNT = 25
    ```
* Output structure:

  ```text
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

## Topics covered

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

## Quality filtering

* Duration: 30 seconds to 60 minutes (configurable)
* Engagement: minimum views, like/comment ratios
* Clickbait detection: filters extreme patterns
* Spam detection: removes spam/low-quality content
* License: Creative Commons by default

## Diversity considerations

* Multi-dimensional search queries (race, gender, age, language)
* Balanced selection across demographics
* Duration distribution target (40% short, 40% medium, 20% long)

## Troubleshooting

* “No videos meet the criteria”

  * Ensure annotated metadata has `Qualitylabel: "Good"`
  * Verify `copyright_notice == "creativeCommon"`
  * Check language fields (`default_language` / `default_audio_language` include `"en"`)
* API rate limiting

  * Increase `rate_limit_delay` in `config.yaml`
  * Process topics in smaller batches
* ffmpeg not found

  * Install via the commands listed under **Prerequisites → Additional requirements**

## Files

* `youtube_metadata_scraper.py` — main YouTube metadata collection
* `parse_topic.py` — download + processing
* `config.yaml` — configuration for both scripts
* `.env` — environment variables (API keys)

## Notes

* Incremental collection is supported (reruns add new videos without duplicates).
* Quality filtering is intended to support research-grade dataset integrity.
* Respect YouTube Terms of Service and copyright laws.
* Video processing is storage/compute intensive; set `MAX_COUNT` accordingly.
