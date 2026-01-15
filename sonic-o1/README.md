# Sonic-O1: Social Non-scripted Interaction Corpus for Fair Multi-form Multimodal Video Understanding

A comprehensive pipeline for creating and evaluating video question answering (VQA) datasets using agentic AI workflows. This repository implements an end-to-end system for curating real-world videos, generating multi-task VQA annotations, and evaluating vision-language models on diverse scenarios.

## Overview

Sonic-O1 provides a systematic approach to building high-quality VQA datasets across 13+ real-world topics including healthcare consultations, job interviews, emergency scenarios, and more. The pipeline leverages state-of-the-art language models (Claude, GPT-4) to generate three types of VQA tasks:

- **Task 1**: Video Summarization (short & detailed summaries with temporal timelines)
- **Task 2**: Multiple Choice Questions (MCQs with distractors)
- **Task 3**: Temporal Localization (finding specific moments in videos)

## Pipeline Architecture

The system is organized into 5 main stages:

```
01_data_curation → 02_caption_generation → 03_demographics_annotation → 04_vqa_generation → 05_evaluation_inference
```

Each stage is self-contained with its own configuration, scripts, and documentation.

## Repository Structure

This repository contains the **pipeline code only**. Dataset and annotations are available separately on HuggingFace.

**Important:** After cloning, you'll have a nested structure: `sonic-o1/sonic-o1/`
- First `sonic-o1/` - The git repository root
- Second `sonic-o1/` - The working directory containing all pipeline code

```
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

**Note:** The following directories are NOT included in this repository and should be downloaded from HuggingFace:
- `dataset/` - Curated videos, audio files, captions, and metadata
- `vqa/` - Generated VQA annotations (3 tasks × 13 topics)

## Quick Start

### Prerequisites

- Python 3.8+
- GPU with CUDA support (for caption generation and inference)
- API keys for:
  - YouTube Data API v3 (only for data curation)
  - Google Gemini API / OpenAI API (for VQA generation)
  - Hugging Face (for model downloads)

### Installation

```bash
# Clone the repository
git clone https://github.com/VectorInstitute/sonic-o1.git

# Navigate to working directory (note the nested structure)
cd sonic-o1/sonic-o1

# Download dataset and VQA annotations from HuggingFace
# Using huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download vector-institute/sonic-o1 --repo-type dataset --local-dir ./

# This will download:
# - dataset/ directory (videos, audios, captions, metadata)
# - vqa/ directory (task annotations)
# Keep the directory names as-is (dataset/ and vqa/)

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

### Environment Setup

Create a `.env` file in each stage directory with required API keys (only for stages you plan to use):

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

**Note:** If you're only evaluating models using the pre-curated dataset, you only need to set up stage 05.

## Dataset Download

Before running the pipeline, download the pre-curated dataset from HuggingFace:

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download the complete dataset
huggingface-cli download vector-institute/sonic-o1 --repo-type dataset --local-dir ./

# The download includes:
# - dataset/videos/           (~XXX GB - video files organized by 13 topics)
# - dataset/audios/           (~XXX GB - extracted audio tracks)
# - dataset/captions/         (~XXX MB - WhisperX transcriptions in SRT format)
# - dataset/*/metadata*.json  (Video metadata with demographics in each topic folder)
# - vqa/task1_summarization/  (~XXX MB - Summaries and timelines, one JSON per topic)
# - vqa/task2_mcq/            (~XXX MB - Multiple choice questions, one JSON per topic)
# - vqa/task3_temporal_localization/ (~XXX MB - Temporal grounding, one JSON per topic)
```

**Important:** Keep the directory names `dataset/` and `vqa/` exactly as downloaded.

**HuggingFace Dataset:** [https://huggingface.co/datasets/vector-institute/sonic-o1](https://huggingface.co/datasets/vector-institute/sonic-o1)

## Pipeline Stages

### 01: Data Curation

Scrapes and filters high-quality YouTube videos based on configurable topics.

```bash
cd 01_data_curation
python parse_topic.py --topics 01_Patient-Doctor_Consultations
```

**Features:**
- Quality-based filtering (engagement, duration, caption availability)
- Metadata extraction (title, description, statistics)
- Deduplication and relevance scoring

**Note:** This stage is for creating new datasets. If using the pre-curated HuggingFace dataset, you can skip this stage.

See [01_data_curation/README.md](01_data_curation/README.md) for details.

### 02: Caption Generation

Generates accurate transcriptions using WhisperX with word-level timestamps.

```bash
cd 02_caption_generation
python whisper_captionGen.py --dataset-root ../dataset --model large-v2
```

**Features:**
- GPU-accelerated transcription
- Word-level alignment and timestamps
- SRT and JSON output formats
- Multi-language support (default: English)

**Note:** Captions are included in the HuggingFace dataset. Run this stage only if you want to regenerate captions or process new videos.

See [02_caption_generation/README.md](02_caption_generation/README.md) for installation and usage.

### 03: Demographics Annotation

Extracts character demographics and interactions from videos using vision-language models.

```bash
cd 03_demographics_annotation
python run_annotation.py --topics 01_Patient-Doctor_Consultations
```

**Features:**
- Character identification (age, gender, ethnicity, role)
- Interaction pattern analysis
- Checkpoint-based processing
- Configurable LLM backends

**Note:** Demographics are included in the HuggingFace dataset metadata. Run this stage only for new videos or re-annotation.

See [03_demographics_annotation/README.md](03_demographics_annotation/README.md) for details.

### 04: VQA Generation

Generates three types of VQA tasks using agentic workflows with Claude/GPT-4.

```bash
cd 04_vqa_generation
python main.py --topics 1,2,3 --tasks summarization,mcq,temporal_localization
```

**Features:**
- **Summarization**: Short bullet points, detailed narratives, temporal timelines
- **MCQ**: Context-aware questions with plausible distractors
- **Temporal Localization**: Event-based timestamp queries
- Parallel processing with retry logic
- Quality validation and checkpointing

**Note:** VQA annotations are included in the HuggingFace dataset under `vqa/` directory. Run this stage only to generate new questions or extend to new topics.

See [04_vqa_generation/README.md](04_vqa_generation/README.md) for configuration options.

### 05: Evaluation & Inference

Evaluates vision-language models on the VQA tasks. **This is the main stage for using the dataset.**

```bash
cd 05_evaluation_inference

# Ensure you've downloaded the dataset from HuggingFace first
python run_evaluation.py \
    --model videollama2 \
    --tasks t1,t2,t3 \
    --topics all \
    --dataset-path ../dataset \
    --vqa-path ../vqa
```

**Supported Models:**
- VideoLLaMA2
- VITA
- Gemini
- GPT
- Uni-MoE variants
- Custom model integration

**Metrics:**
- Task 1: ROUGE, METEOR, BERTScore
- Task 2: Accuracy, F1-score
- Task 3: Temporal IoU, Precision@K

See [05_evaluation_inference/README.md](05_evaluation_inference/README.md) for model setup and metrics.

## Dataset Topics

The dataset covers 13 diverse real-world scenarios:

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

Each topic contains 15-25 carefully curated videos with complete annotations.

## Output Format Examples

### Task 1: Summarization

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

### Task 2: MCQ

```json
{
  "video_id": "abc123",
    "question": "...",
    "options": [
    "(A) It provides a detailed ....",
    "(B) It helps ...",
    "(C) It ensures ....",
    "(D) It categorizes ...",
    "(E) Not enough evidence"
    ],
    "answer_index": 1,
    "answer_letter": "B",
    "rationale": "..."


}
```

### Task 3: Temporal Localization

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
        "answer": {
        "start_s": 35.0,
        "end_s": 36.62
        },
    },
  ]
}
```

## Configuration

Each stage uses YAML configuration files:

- `01_data_curation/config.yaml` - Search and filtering parameters
- `02_caption_generation/config_whisper.yaml` - Transcription settings
- `03_demographics_annotation/config.yaml` - LLM and annotation config
- `04_vqa_generation/config/*.yaml` - Task-specific VQA generation
- `05_evaluation_inference/configs/*.yaml` - Model and metric settings

## Using the Dataset

### Quick Start for Evaluation

If you just want to evaluate models on the dataset:

```bash
# 1. Clone this repository
git clone https://github.com/VectorInstitute/sonic-o1.git

# 2. Navigate to working directory
cd sonic-o1/sonic-o1

# 3. Download dataset from HuggingFace
pip install huggingface_hub
huggingface-cli download vector-institute/sonic-o1 --repo-type dataset --local-dir ./

# 3. Install base dependencies
pip install -r requirements_venv.txt (or uv sync)

# 4. Install model-specific dependencies (see 05_evaluation_inference/README.md)
cd 05_evaluation_inference
pip install -r models_requirements/videollama2_requirements.txt

# 5. Run evaluation
python run_evaluation.py --model videollama2 --tasks t1,t2,t3 --topics all
```

### Creating Your Own Dataset

To create a new dataset or extend the existing one:

```bash
# Stage 1: Collect videos for specific topics
cd 01_data_curation
python parse_topic.py --topics 01,02,03

# Stage 2: Generate captions for collected videos
cd ../02_caption_generation
python whisper_captionGen.py --topics 01_Patient-Doctor_Consultations

# Stage 3: Annotate demographics
cd ../03_demographics_annotation
python run_annotation.py --topics 01_Patient-Doctor_Consultations

# Stage 4: Generate VQA tasks
cd ../04_vqa_generation
python main.py --topics 1 --task summarization

# Stage 5: Evaluate models
cd ../05_evaluation_inference
python run_evaluation.py --model videollama2 --tasks t1
```

### Adding New Topics

1. Add topic definition to `01_data_curation/config.yaml`
2. Run data curation with the new topic
3. Process through remaining pipeline stages
4. Update topic mappings in stage 4 and 5 configs

### Adding New Models

1. Create model wrapper in `05_evaluation_inference/models/`
2. Add model configuration to `05_evaluation_inference/configs/models_config.yaml`
3. Implement required inference methods
4. Run evaluation with `--model your_model_name`

## Citation

If you use this dataset or pipeline in your research, please cite:

```bibtex
@article{sonic-o1-2025,
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WhisperX for transcription capabilities
- Google Gemini and OpenAI GPT-4 for VQA generation
- Open-source VLM community for evaluation frameworks

## Data Availability

### HuggingFace Dataset

The complete dataset (videos, captions, annotations) is available at:
**[https://huggingface.co/datasets/vector-institute/sonic-o1](https://huggingface.co/datasets/vector-institute/sonic-o1)**

**Dataset Structure (after downloading):**
```
sonic-o1/
├── dataset/
│   ├── videos/                    # Video files organized by 13 topics
│   │   ├── 01_Patient-Doctor_Consultations/
│   │   │   ├── video_001.mp4
│   │   │   └── metadata_enhanced.json  # With demographics
│   │   └── ...
│   ├── audios/                    # Extracted audio tracks (same structure)
│   └── captions/                  # WhisperX transcriptions (SRT format)
│       ├── 01_Patient-Doctor_Consultations/
│       │   ├── caption_001.srt
│       │   └── ...
│       └── ...
└── vqa/
    ├── task1_summarization/       # One JSON file per topic
    │   ├── 01_Patient-Doctor_Consultations.json
    │   └── ...
    ├── task2_mcq/                 # One JSON file per topic
    │   ├── 01_Patient-Doctor_Consultations.json
    │   └── ...
    └── task3_temporal_localization/  # One JSON file per topic
        ├── 01_Patient-Doctor_Consultations.json
        └── ...
```

**Important:** Keep the directory names `dataset/` and `vqa/` exactly as downloaded.

### GitHub Repository

Pipeline code is available at:
**[https://github.com/VectorInstitute/sonic-o1](https://github.com/VectorInstitute/sonic-o1)**

Clone and navigate:
```bash
git clone https://github.com/VectorInstitute/sonic-o1.git
cd sonic-o1/sonic-o1  # Note: repository contains sonic-o1 subdirectory
```

## Troubleshooting

### Common Issues

1. **Disk Quota Exceeded**: Set cache directories to scratch space (see [02_caption_generation/README.md](02_caption_generation/README.md))
2. **API Rate Limits**: Adjust `rate_limit_delay` in configs
3. **CUDA OOM**: Use smaller models or reduce batch sizes
4. **Missing Dependencies**: Check individual stage README files
5. **Dataset Path Issues**: Ensure you're in the `sonic-o1/sonic-o1` directory after cloning

### Support

For questions and issues:
- Open an issue on [GitHub](https://github.com/VectorInstitute/sonic-o1/issues)
- Check individual stage README files for detailed troubleshooting
- Review configuration examples in stage-specific `config/` directories

