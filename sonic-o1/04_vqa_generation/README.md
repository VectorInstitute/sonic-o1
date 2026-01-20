# Video Question-Answer (VQA) Generation

This directory handles automatic generation of three video QA-related tasks using Gemini-based multimodal models:
1. **Summarization** - Short + detailed summaries
2. **Multiple Choice Questions (MCQ)** - Segment-level questions with options
3. **Temporal Localization** - Segment-level time-related questions

## Prerequisites

Before running this step, you must have completed:
1. **Data Curation** (see [01_data_curation](../01_data_curation/))
2. **Caption Generation** (see [02_caption_generation](../02_caption_generation/))
3. **Demographics Annotation** (see [03_demographics_annotation](../03_demographics_annotation/))

Your `dataset/` directory should have:
```

dataset/
├── videos/
│   ├── 01_<Topic_Name>/
│   │   ├── video_001.mp4
│   │   ├── video_002.mp4
│   │   └── metadata_enhanced.json
│   ├── 02_<Topic_Name>/
│   │   └── ...
│   └── ...
├── audios/
│   ├── 01_<Topic_Name>/
│   │   ├── audio_001.m4a
│   │   └── ...
│   └── ...
└── captions/
├── 01_<Topic_Name>/
│   ├── caption_001.srt
│   └── ...
└── ...

````

## Required Packages

All required Python packages are included in [requirements_venv.txt](../../requirements_venv.txt), including:
- `google-generativeai`
- `openai`
- `python-dotenv`
- `pyyaml`
- `tqdm`

## API Setup

### Gemini API Key (Required)
1. Go to Google AI Studio
2. Generate an API key

### OpenAI API Key (Optional)
Required only if your temporal pipeline uses OpenAI-based judging/validation.

### Set API Keys
Create a `.env` file in this directory:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
````

## Configuration

Edit [config/vqa_config.yaml](config/vqa_config.yaml) to customize processing settings.

### Model Settings

```yaml
gemini:
  model_name: "gemini-2.5-flash"
  temperature: 0.3
  max_output_tokens: 2048
```

### Rate Limiting

```yaml
rate_limit:
  delay_between_videos: 45
  delay_after_long_video: 60
```

## Usage

Run from the project root so relative paths resolve correctly.

### Process All Topics - All Tasks

```bash
python 04_vqa_generation/main.py --all
```

### Process Specific Topics

```bash
python 04_vqa_generation/main.py --topics 1,2,3
python 04_vqa_generation/main.py --topics 5
```

### Process Specific Task Only

```bash
python 04_vqa_generation/main.py --topics 1,2 --task summarization
python 04_vqa_generation/main.py --all --task mcq
python 04_vqa_generation/main.py --all --task temporal
```

### Fill Empty Demographics

```bash
python 04_vqa_generation/fill_empty_demographics.py --dry-run
python 04_vqa_generation/fill_empty_demographics.py
python 04_vqa_generation/fill_empty_demographics.py --topics 10,11
```

### Standardize Demographics

```bash
python 04_vqa_generation/standardize_demographics.py --dry-run
python 04_vqa_generation/standardize_demographics.py
python 04_vqa_generation/standardize_demographics.py --topics 1,2,3
```

## Output Structure

Outputs are written to the configured output directory (default: `/vqa/`) in per-task folders, with one JSON file per topic:

```
<output_dir>/
├── task1_summarization/
│   ├── 01_<Topic_Name>.json
│   ├── 02_<Topic_Name>.json
│   └── ...
├── task2_mcq/
│   ├── 01_<Topic_Name>.json
│   ├── 02_<Topic_Name>.json
│   └── ...
└── task3_temporal_localization/
    ├── 01_<Topic_Name>.json
    ├── 02_<Topic_Name>.json
    └── ...
```

Each output file has a shared wrapper:

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
    "..."
  ],
  "summary_detailed": "...",
  "confidence": 0.92
}
```

### Task 2: MCQ (`task2_mcq/*.json`)

`entries` is a list with one entry per generated question (typically multiple per video). Segment fields are used for merging/replacement:

```json
{
  "video_id": "001",
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "segment": {
    "start": 120.0,
    "end": 180.0
  },
  "question": "...",
  "options": ["...", "...", "...", "...", "..."],
  "correct_answer": 0,
  "confidence": 0.85
}
```

### Task 3: Temporal Localization (`task3_temporal_localization/*.json`)

`entries` is a list with one entry per generated temporal question. Segment fields are used for merging/replacement:

```json
{
  "video_id": "001",
  "topic_id": 1,
  "topic_name": "Patient-Doctor Consultations",
  "segment": {
    "start": 45.0,
    "end": 90.0
  },
  "question": "...",
  "answer": "...",
  "confidence": 0.81
}
```

## Processing Pipeline

1. **Generate VQA**

```bash
python 04_vqa_generation/main.py --all
```

2. **Fill Empty Demographics**

```bash
python 04_vqa_generation/fill_empty_demographics.py
```

3. **Standardize Demographics**

```bash
python 04_vqa_generation/standardize_demographics.py
```

## Directory Structure

```
04_vqa_generation/
├── config/
│   └── vqa_config.yaml
├── models/
│   ├── base_gemini.py
│   ├── summarization_model.py
│   ├── mcq_model.py
│   └── temporal_localization_model.py
├── prompts/
│   ├── summarization_prompts.py
│   ├── mcq_prompts.py
│   └── temporal_localization_prompts.py
├── utils/
├── main.py
├── fill_empty_demographics.py
├── standardize_demographics.py
└── .env
```


## Notes

- The script skips videos that already have VQA generated (configurable)
- Raw API responses are saved if `save_raw_responses: true` in config
- Temporal localization uses GPT-4V for validation (optional, requires OpenAI key)
- All paths are relative to project root, so always run from sonic-o1 directory
- Demographics optimization: Task 2 reuses Task 3 demographics (same segments), Task 1 generates separately
- Check scripts (check_empty_demographics.py, check_failed_summary.py) are helper utilities not in git
