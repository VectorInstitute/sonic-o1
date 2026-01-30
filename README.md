# SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/vector-institute/sonic-o1)
[![Leaderboard](https://img.shields.io/badge/HuggingFace-Leaderboard-blue?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/vector-institute/sonic-o1-leaderboard)
[![arXiv](https://img.shields.io/badge/arXiv-2601.21666-b31b1b.svg)](https://arxiv.org/abs/2601.21666)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-lightgrey)](#citation)
[![Contact](https://img.shields.io/badge/Contact-shaina.raza%40vectorinstitute.ai-green)](mailto:shaina.raza@vectorinstitute.ai)



A comprehensive pipeline for creating and evaluating video question answering (VQA) datasets using agentic AI workflows. This repository implements an end-to-end system for curating real-world videos, generating multi-task VQA annotations, and evaluating vision-language models on diverse scenarios.

## Overview

SONIC-O1 provides a systematic approach to building high-quality VQA datasets across 13+ real-world topics including healthcare consultations, job interviews, emergency scenarios, and more. The pipeline generates three types of VQA tasks (via state-of-the-art LLMs such as Gemini and GPT-4):

- **Task 1 — Summarization:** Short + detailed summaries with temporal timelines
- **Task 2 — Multiple Choice (MCQ):** Questions with plausible distractors
- **Task 3 — Temporal Localization:** Finding specific moments in videos

## Pipeline Architecture

The system is organized into 5 stages:

```text
01_data_curation → 02_caption_generation → 03_demographics_annotation → 04_vqa_generation → 05_evaluation_inference
````

Each stage is self-contained with its own configuration, scripts, and documentation.

## Repository Structure

This repository contains the **pipeline code only**. The dataset and annotations are available separately on Hugging Face (see links at top).

**Important:** After cloning, you'll have a nested structure: `sonic-o1/sonic-o1/`

* First `sonic-o1/` — the git repository root
* Second `sonic-o1/` — the working directory containing all pipeline code

```text
sonic-o1/                          # Git repository root
└── sonic-o1/                      # Working directory (cd here to run commands)
    ├── 01_data_curation/          # YouTube video collection and filtering
    ├── 02_caption_generation/     # WhisperX-based transcription
    ├── 03_demographics_annotation/# Character demographics extraction
    ├── 04_vqa_generation/         # Multi-task VQA generation
    ├── 05_evaluation_inference/   # Model evaluation framework
    ├── dataset/                   # Downloaded from HuggingFace
    └── vqa/                       # Downloaded from HuggingFace
```

**Not included in this repo (download from Hugging Face):**

* `dataset/` — curated videos, audio, captions, and metadata
* `vqa/` — generated VQA annotations (3 tasks × topics)

## Quick Start

### Prerequisites

* Python 3.8+
* GPU with CUDA support (recommended for caption generation and inference)
* API keys/tokens (only for the stages you plan to run):

  * YouTube Data API v3 (Stage 01; only if collecting new videos)
  * Google Gemini API / OpenAI API (Stages 03–05 depending on backend)
  * Hugging Face token (Stage 05 model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/VectorInstitute/sonic-o1.git

# Navigate to working directory (note the nested structure)
cd sonic-o1/sonic-o1

# (Recommended) Download dataset + VQA annotations from Hugging Face
pip install huggingface_hub
huggingface-cli download vector-institute/sonic-o1 --repo-type dataset --local-dir ./

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install base dependencies
pip install -r requirements_venv.txt
# Or using pyproject.toml:
pip install -e .

# Note: Each stage (01-05) may have additional dependencies.
# Stage 05 has model-specific requirements in 05_evaluation_inference/models_requirements/
```

### Environment Setup (API Keys)

Create a `.env` file in each stage directory **only if you plan to run that stage**:

```bash
# 01_data_curation/.env (only if collecting new videos)
YOUTUBE_API_KEY=your_youtube_api_key

# 03_demographics_annotation/.env (only if annotating new videos)
GEMINI_API_KEY=your_gemini_api_key

# 04_vqa_generation/.env (only if generating new VQA tasks)
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key

# 05_evaluation_inference/.env (for model evaluation)
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key  # if using Gemini models
OPENAI_API_KEY=your_openai_api_key  # if using OpenAI models
```

**Note:** If you're only evaluating models using the pre-curated dataset, you typically only need Stage 05.

## Dataset Download

Before running the pipeline, download the pre-curated dataset from Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli download vector-institute/sonic-o1 --repo-type dataset --local-dir ./
```

The download includes:

* `dataset/videos/` (topic-organized video files)
* `dataset/audios/` (extracted audio)
* `dataset/captions/` (WhisperX transcriptions)
* per-topic metadata JSON files
* `vqa/` directory (task annotations)

**Important:** Keep the directory names `dataset/` and `vqa/` exactly as downloaded.

## Pipeline Stages (what to run)

### 01 — Data Curation (optional)

Scrapes and filters high-quality YouTube videos based on configurable topics.

```bash
cd 01_data_curation
python parse_topic.py --topics 01_Patient-Doctor_Consultations
```

See `01_data_curation/README.md` for details.

### 02 — Caption Generation (optional)

Generates transcriptions using WhisperX with word-level timestamps.

```bash
cd 02_caption_generation
python whisper_captionGen.py --dataset-root ../dataset --model large-v2
```

See `02_caption_generation/README.md` for installation and usage.

### 03 — Demographics Annotation (optional)

Extracts character demographics and interactions using vision-language models.

```bash
cd 03_demographics_annotation
python run_annotation.py --topics 01_Patient-Doctor_Consultations
```

See `03_demographics_annotation/README.md` for details.

### 04 — VQA Generation (optional)

Generates VQA tasks using agentic workflows with Gemini/OpenAI backends.

```bash
cd 04_vqa_generation
python main.py --topics 1,2,3 --tasks summarization,mcq,temporal_localization
```

See `04_vqa_generation/README.md` for configuration options.

### 05 — Evaluation & Inference (main entrypoint)

Evaluates vision-language models on the VQA tasks.

```bash
cd 05_evaluation_inference

python run_evaluation.py \
  --model videollama2 \
  --tasks t1,t2,t3 \
  --topics all \
  --dataset-path ../dataset \
  --vqa-path ../vqa
```

Supported models include: VideoLLaMA2, VITA, Gemini, GPT, Uni-MoE variants, and custom integrations.

Metrics:

* Task 1: ROUGE-L, Judge-Score
* Task 2: Accuracy
* Task 3: Temporal IoU, Precision@K, MAE

See `05_evaluation_inference/README.md` for model setup and metrics.

## Dataset Topics

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
13. Olympics (Sports events)

Each topic contains 15–25 carefully curated videos with complete annotations.

## Output Format Examples

### Task 1 — Summarization

```json
{
  "video_id": "abc123",
  "summary_short": ["• Bullet point 1", "• Bullet point 2"],
  "summary_detailed": "Comprehensive narrative...",
  "timeline": [
    {
      "start": "00:01:23",
      "end": "00:02:45",
      "title": "Section Title",
      "note": "Description of events"
    }
  ]
}
```

### Task 2 — MCQ

```json
{
  "video_id": "abc123",
  "question": "...",
  "options": [
    "(A) ...",
    "(B) ...",
    "(C) ...",
    "(D) ...",
    "(E) Not enough evidence"
  ],
  "answer_index": 1,
  "answer_letter": "B",
  "rationale": "..."
}
```

### Task 3 — Temporal Localization

```json
{
  "video_id": "abc123",
  "questions": [
    {
      "question_id": "001",
      "question": "After the speaker ...",
      "temporal_relation": "after",
      "anchor_event": "The speaker ..",
      "target_event": "The speaker states that he is a ...",
      "answer": { "start_s": 35.0, "end_s": 36.62 }
    }
  ]
}
```

## Configuration

Each stage uses YAML configuration files:

* `01_data_curation/config.yaml` — search + filtering parameters
* `02_caption_generation/config_whisper.yaml` — transcription settings
* `03_demographics_annotation/config.yaml` — LLM + annotation settings
* `04_vqa_generation/config/*.yaml` — task-specific VQA generation
* `05_evaluation_inference/configs/*.yaml` — model + metric settings

## Citation

If you use this dataset or pipeline in your research, please cite:

```bibtex
@article{radwan2026sonico1,
  title={SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding},
  author={Radwan, Ahmed Y and Emmanouilidis, Christos and Tabassum, Hina and Pandya, Deval and Raza, Shaina},
  journal={arXiv preprint arXiv:2601.21666},
  year={2026}
}
```

## License

This dataset is licensed under the **Vector Institute License**. The SONIC-O1 dataset may only be accessed and used by:

* Academic entities for non-commercial academic research purposes
* Vector Institute sponsors and partners

By accessing or using this dataset, you agree to be bound by the terms of the Vector Institute License.

**Attribution Required:**
When using this dataset, include:
"This work is licensed under the Vector Institute License, Copyright © Vector Institute. All Rights Reserved."

For products or services built using this dataset, prominently display:
**"Built with Vector Institute SONIC-O1"**

## Acknowledgments

Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

This research was funded by the European Union's Horizon Europe research and innovation programme under the **AIXPERT project** (Grant Agreement No. 101214389).

## Troubleshooting

Common issues:

1. Disk quota exceeded: set cache directories to scratch space (see `02_caption_generation/README.md`)
2. API rate limits: adjust `rate_limit_delay` in configs
3. CUDA OOM: use smaller models or reduce batch sizes
4. Missing dependencies: check individual stage README files
5. Dataset path issues: ensure you're in `sonic-o1/sonic-o1` after cloning

## Support

* Open an issue on GitHub: [https://github.com/VectorInstitute/sonic-o1/issues](https://github.com/VectorInstitute/sonic-o1/issues)
* Check individual stage README files for detailed troubleshooting
* Review stage-specific configuration examples in `config/` directories

