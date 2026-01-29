# Evaluation & Inference Pipeline

## Overview

This directory handles model evaluation and inference for video question-answering tasks. It supports multiple open-source and commercial models with proper environment management, inference execution, and comprehensive metrics computation.

### Directory Structure

```
05_evaluation_inference/
├── run_evaluation.py          # Main evaluation pipeline orchestrator
├── README.md                  # This file
│
├── inference/                 # Inference execution
│   └── run_inference.py      # Standalone inference script
│
├── models/                    # Model implementations
│   ├── base_model.py         # Base class for all models
│   ├── gemini.py             # Google Gemini API
│   ├── gpt4o.py              # OpenAI GPT-4o API
│   ├── qwen3.py              # Qwen3 VL model
│   ├── minicpm.py            # MiniCPM-V model
│   ├── phi4.py               # Phi-4 Vision model
│   ├── unimoe.py             # Uni-MoE model
│   ├── videollama.py         # Video-LLaMA model
│   └── vita.py               # VITA model
│
├── metrics/                   # Metrics computation
│   ├── compute_metrics.py    # Main metrics computation
│   ├── t1_metrics.py         # Task 1 (Summarization) metrics
│   ├── t2_metrics.py         # Task 2 (MCQ) metrics
│   ├── t3_metrics.py         # Task 3 (Temporal) metrics
│   ├── llm_judge_gpt.py      # GPT-based LLM judge
│   └── llm_judge_qwen.py     # Qwen-based LLM judge
│
├── prompts/                   # Task-specific prompts
│   ├── t1_prompts.py         # Task 1 prompts
│   ├── t2_prompts.py         # Task 2 prompts
│   └── t3_prompts.py         # Task 3 prompts
│
├── utils/                     # Utility functions
│   ├── audio_processor.py    # Audio extraction and processing
│   ├── caption_handler.py    # Caption/subtitle processing
│   ├── config_loader.py      # Configuration management
│   ├── frame_sampler.py      # Video frame sampling strategies
│   ├── mm_process_pyav.py   # Multimedia processing with PyAV
│   └── segmenter.py          # Video segmentation utilities
│
├── models_config.yaml         # Model and evaluation configuration
│
├── models_requirements/        # Model-specific requirements
│   ├── requirements_venv_llama.txt
│   ├── requirements_venv_minicpm.txt
│   ├── requirements_venv_phi4.txt
│   ├── requirements_venv_unimoe.txt
│   ├── requirements_venv_vita.txt
│   └── requirements_qwen3.txt
│
├── external_repos/            # External model repositories
│   └── README.md             # Details on included repos
│
└── results/                   # Output directory
    ├── predictions/           # Model predictions
    └── scores/               # Evaluation scores
```

### Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Model Inference                                             │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  dataset/videos/<topic>/video_*.mp4                          │
│         dataset/audios/<topic>/audio_*.m4a                          │
│         dataset/captions/<topic>/caption_*.srt                      │
│         vqa/task*/<topic>.json (ground truth)                        │
│ Output: results/predictions/<model>/<task>/<topic>.json             │
│         ├── Model predictions for each VQA entry                     │
│         └── Per-task, per-topic prediction files                    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Metrics Computation                                         │
│ ──────────────────────────────────────────────────────────────────  │
│ Input:  results/predictions/<model>/<task>/<topic>.json             │
│         vqa/task*/<topic>.json (ground truth)                        │
│ Output: results/scores/<model>/<task>/<topic>_scores.json          │
│         ├── Task-specific metrics (BLEU, ROUGE, accuracy, etc.)    │
│         ├── LLM judge scores (if enabled)                            │
│         └── Aggregated evaluation results                            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Results Analysis                                            │
│ ──────────────────────────────────────────────────────────────────  │
│ • Review prediction files in results/predictions/                     │
│ • Analyze metric scores in results/scores/                          │
│ • Compare model performance across tasks                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Supported Models

The pipeline supports multiple video understanding models:

**API-Based Models:**
- **Gemini** - Google Gemini multimodal API
- **GPT-4o** - OpenAI GPT-4o Vision API

**Open-Source Models:**
- **Qwen3** - Qwen3 VL model
- **MiniCPM-V** - MiniCPM Vision model
- **Phi-4** - Phi-4 Vision model
- **Uni-MoE** - Uni-MoE multimodal model
- **Video-LLaMA** - Video-LLaMA2 model
- **VITA** - VITA video understanding model

### Features

- **Multiple Model Support**: Evaluate various open-source and commercial models
- **Three Task Types**: Task 1 (Summarization), Task 2 (MCQ), Task 3 (Temporal Localization)
- **Flexible Environment Management**: Model-specific virtual environments for compatibility
- **Comprehensive Metrics**: Task-specific metrics plus LLM judge evaluation
- **Resume Capability**: Skip already processed entries and resume from checkpoints
- **Batch Processing**: Process multiple models, tasks, and topics in one run
- **Error Handling**: Retry logic with fallback strategies for processing failures

## Prerequisites

Before running this step, you must have completed:

1. **Data Curation** (see [01_data_curation](../01_data_curation/))
   - Downloaded videos and audio files
   - Generated metadata.json files

2. **Caption Generation** (see [02_caption_generation](../02_caption_generation/))
   - Generated captions for all videos (SRT format)

3. **Demographics Annotation** (see [03_demographics_annotation](../03_demographics_annotation/))
   - Generated metadata_enhanced.json files with demographics

4. **VQA Generation** (see [04_vqa_generation](../04_vqa_generation/))
   - Generated VQA ground truth files in `vqa/` directory:
     - `vqa/task1_summarization/<topic>.json`
     - `vqa/task2_mcq/<topic>.json`
     - `vqa/task3_temporal_localization/<topic>.json`

Your `dataset/` and `vqa/` directories should have this structure:
```
dataset/
├── videos/<topic_name>/
│   ├── video_001.mp4
│   └── metadata_enhanced.json
├── audios/<topic_name>/
│   └── audio_001.m4a
└── captions/<topic_name>/
    └── caption_001.srt

vqa/
├── task1_summarization/
│   └── 01_Patient-Doctor_Consultations.json
├── task2_mcq/
│   └── 01_Patient-Doctor_Consultations.json
└── task3_temporal_localization/
    └── 01_Patient-Doctor_Consultations.json
```

5. **API Setup** (for API-based models)
   - **Gemini API Key** (for Gemini model)
     - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
     - Set in `.env` file: `GEMINI_API_KEY=your_key_here`
   - **OpenAI API Key** (for GPT-4o model)
     - Get from [OpenAI Platform](https://platform.openai.com/api-keys)
     - Set in `.env` file: `OPENAI_API_KEY=your_key_here`

## Installation

### 1. General Environment Setup

Most models use the general project environment. Install dependencies from the project root:

```bash
# From the project root (parent of sonic-o1)
cd /path/to/VideoQA-Agentic
source .venv/bin/activate
pip install -e .
# or
pip install -r requirements.txt
```

### 2. Model-Specific Environments

Some models require specialized dependencies. Only install these if you plan to use the specific models:

**Video-LLaMA:**
```bash
cd /path/to/sonic-o1/05_evaluation_inference
python -m venv venv_llama
source venv_llama/bin/activate
pip install -r models_requirements/requirements_venv_llama.txt
```

**MiniCPM-V:**
```bash
python -m venv venv_minicpm
source venv_minicpm/bin/activate
pip install -r models_requirements/requirements_venv_minicpm.txt
```

**Phi-4 Vision:**
```bash
python -m venv venv_phi4
source venv_phi4/bin/activate
pip install -r models_requirements/requirements_venv_phi4.txt
```

**Uni-MoE:**
```bash
python -m venv venv_unimoe
source venv_unimoe/bin/activate
pip install -r models_requirements/requirements_venv_unimoe.txt
```

**VITA:**
```bash
python -m venv venv_vita
source venv_vita/bin/activate
pip install -r models_requirements/requirements_venv_vita.txt
```

**Qwen3:**
```bash
# Uses general environment, but may need:
pip install -r models_requirements/requirements_qwen3.txt
```

### 3. External Repositories

Some models require external repositories with compatibility fixes. See [external_repos/README.md](external_repos/README.md) for setup instructions.

## Configuration

Edit [models_config.yaml](models_config.yaml) to customize evaluation settings.

### Dataset Paths
```yaml
dataset_path: "dataset"
vqa_path: "vqa"

results:
  predictions_path: "05_evaluation_inference/results/predictions"
  scores_path: "05_evaluation_inference/results/scores"
```

### Tasks and Topics
```yaml
tasks:
  - task1_summarization
  - task2_mcq
  - task3_temporal_localization

topics:
  - "01_Patient-Doctor_Consultations"
  - "02_Job_Interviews"
  # ... (all 13 topics)
```

### Preprocessing Settings
```yaml
preprocessing:
  t2_t3:
    segment_max_duration: 180      # Max segment duration (seconds)
    image_model_frames: 128         # Frames for image-based models
    video_model_fps: 1              # FPS for video models
```

### Retry Logic
```yaml
retry:
  max_attempts: 4
  fps_fallback: [1, 0.5, 0.25]
  frame_count_fallback: [256, 128, 64, 32]
  audio_chunks_fallback: [null, 64, 32, 16]
  audio_chunk_duration_sec: 10.0
```

### Metrics Configuration
```yaml
metrics:
  llm_judge_model: "gpt-5-mini"    # or "Qwen/Qwen3-8B"
  # Task-specific metric settings...
```

### Model-Specific Configuration

Each model has its own configuration section in `models_config.yaml`:
```yaml
models:
  gemini:
    class: "Gemini"
    api_key: "${GEMINI_API_KEY}"
    # ... model-specific settings
  
  videollama:
    class: "VideoLLaMA2"
    model_path: "/path/to/model"
    # ... model-specific settings
```

## Usage

**IMPORTANT**: Always run scripts from the `sonic-o1` directory (parent directory), not from within `05_evaluation_inference`. This is required for relative paths to work correctly.

### Full Evaluation Pipeline

Run both inference and metrics computation:

```bash
# Navigate to sonic-o1 directory
cd /path/to/sonic-o1

# Activate appropriate environment
source .venv/bin/activate  # or source venv_<model>/bin/activate

# Run full evaluation for one model
python 05_evaluation_inference/run_evaluation.py \
    --model gemini \
    --tasks all

# Run for specific tasks
python 05_evaluation_inference/run_evaluation.py \
    --model gpt4o \
    --tasks t1 t2

# Run for specific topics
python 05_evaluation_inference/run_evaluation.py \
    --model qwen3 \
    --topics "01_Patient-Doctor_Consultations" "02_Job_Interviews"
```

### Inference Only

Run inference without computing metrics:

```bash
python 05_evaluation_inference/run_evaluation.py \
    --model gemini \
    --tasks all \
    --inference-only
```

### Metrics Only

Compute metrics on existing predictions:

```bash
python 05_evaluation_inference/run_evaluation.py \
    --model gemini \
    --metrics-only
```

### Multiple Models

Evaluate multiple models in sequence:

```bash
python 05_evaluation_inference/run_evaluation.py \
    --models gemini gpt4o qwen3 \
    --tasks all
```

### Standalone Inference

Run inference directly without the orchestrator:

```bash
python 05_evaluation_inference/inference/run_inference.py \
    --model gemini \
    --tasks task1_summarization task2_mcq \
    --topics "01_Patient-Doctor_Consultations" \
    --config 05_evaluation_inference/models_config.yaml
```

### Standalone Metrics

Compute metrics directly:

```bash
python 05_evaluation_inference/metrics/compute_metrics.py \
    --predictions results/predictions/gemini/task1_summarization/01_Patient-Doctor_Consultations.json \
    --ground_truth vqa/task1_summarization/01_Patient-Doctor_Consultations.json \
    --task t1
```

### Command-Line Arguments

The main evaluation script supports many options:

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Single model name to evaluate | `--model gemini` |
| `--models` | Multiple model names | `--models gemini gpt4o` |
| `--config` | Path to config file | `--config custom_config.yaml` |
| `--tasks` | Tasks to evaluate (t1, t2, t3, or 'all') | `--tasks t1 t2` |
| `--topics` | Topics to evaluate | `--topics "01_Patient-Doctor_Consultations"` |
| `--inference-only` | Run inference only, skip metrics | `--inference-only` |
| `--metrics-only` | Compute metrics only, skip inference | `--metrics-only` |
| `--skip-existing` | Skip already processed entries (default: True) | `--skip-existing` |
| `--force-rerun` | Force re-run even if outputs exist | `--force-rerun` |
| `--retry-failed` | Retry only failed entries | `--retry-failed` |
| `--experiment-name` | Experiment name for organizing results | `--experiment-name "frames_16"` |
| `--no-llm-judge` | Skip LLM judge evaluation | `--no-llm-judge` |
| `--dataset-path` | Override dataset path | `--dataset-path custom_dataset/` |
| `--vqa-path` | Override VQA path | `--vqa-path custom_vqa/` |

**Examples:**

```bash
# Full evaluation with experiment name
python 05_evaluation_inference/run_evaluation.py \
    --model gemini \
    --tasks all \
    --experiment-name "modality_audio_only"

# Retry failed entries only
python 05_evaluation_inference/run_evaluation.py \
    --model videollama \
    --tasks t2 \
    --retry-failed

# Force rerun with custom paths
python 05_evaluation_inference/run_evaluation.py \
    --model gpt4o \
    --tasks t1 \
    --force-rerun \
    --dataset-path custom_dataset/ \
    --vqa-path custom_vqa/
```

## Output

The pipeline creates organized output files in the `results/` directory.

### Output Location
```
results/
├── predictions/
│   └── <model>/
│       ├── task1_summarization/
│       │   ├── 01_Patient-Doctor_Consultations.json
│       │   └── ...
│       ├── task2_mcq/
│       │   └── ...
│       └── task3_temporal_localization/
│           └── ...
│
└── scores/
    └── <model>/
        ├── task1_summarization/
        │   ├── 01_Patient-Doctor_Consultations_scores.json
        │   └── ...
        ├── task2_mcq/
        │   └── ...
        └── task3_temporal_localization/
            └── ...
```

### Prediction Format

Predictions follow the same structure as ground truth VQA files, with model predictions added:

**Task 1 (Summarization):**
```json
{
  "video_id": "001",
  "prediction": {
    "summary_short": ["...", "..."],
    "summary_detailed": "...",
    "timeline": [...],
    "glossary": [...]
  },
  "ground_truth": { ... }
}
```

**Task 2 (MCQ):**
```json
{
  "video_id": "001",
  "segment": {"start": 120.0, "end": 180.0},
  "prediction": {
    "answer": 0,
    "explanation": "..."
  },
  "ground_truth": { ... }
}
```

**Task 3 (Temporal):**
```json
{
  "video_id": "001",
  "segment": {"start": 45.0, "end": 90.0},
  "prediction": {
    "answer": "...",
    "temporal_relation": "after"
  },
  "ground_truth": { ... }
}
```

### Metrics Format

Scores files contain comprehensive evaluation metrics:

```json
{
  "model": "gemini",
  "task": "task1_summarization",
  "topic": "01_Patient-Doctor_Consultations",
  "metrics": {
    "bleu": 0.45,
    "rouge_l": 0.52,
    "rouge_1": 0.58,
    "rouge_2": 0.41,
    "llm_judge_score": 0.78
  },
  "num_entries": 25,
  "computed_at": "2026-01-14 12:34:56"
}
```

## Processing Time

Processing time varies significantly by model and task:

- **API Models (Gemini, GPT-4o)**: ~5-30 seconds per entry
- **Open-Source Models**: ~10-120 seconds per entry (depends on GPU)
- **Task 1 (Summarization)**: Faster (single summary per video)
- **Task 2 (MCQ)**: Medium (multiple questions per video)
- **Task 3 (Temporal)**: Slower (multiple questions with temporal reasoning)

### Estimated Time for Full Dataset

- **13 topics × 25 videos = 325 videos**
- **API models**: ~2-4 hours (all tasks)
- **Open-source models**: ~4-12 hours (all tasks, GPU-dependent)

## Troubleshooting

### Environment Issues

**Problem**: `ModuleNotFoundError` or import errors

**Solution**: Ensure you're using the correct environment:
```bash
# Check which environment is active
which python

# Activate correct environment
source .venv/bin/activate  # General environment
# or
source venv_<model>/bin/activate  # Model-specific environment
```

### Path Resolution Errors

**Problem**: `FileNotFoundError` for dataset or VQA files

**Solution**: Always run from the `sonic-o1` directory:
```bash
cd /path/to/sonic-o1
python 05_evaluation_inference/run_evaluation.py --model gemini
```

### API Key Not Found

**Problem**: `ERROR: API key not set!` (for Gemini/GPT-4o)

**Solution**:
```bash
# Create .env file in project root
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "OPENAI_API_KEY=your_key_here" >> .env

# Or export environment variables
export GEMINI_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

### GPU Out of Memory

**Problem**: `CUDA out of memory` (for open-source models)

**Solution**: Adjust preprocessing settings in `models_config.yaml`:
```yaml
preprocessing:
  t2_t3:
    image_model_frames: 64  # Reduce from 128
    video_model_fps: 0.5     # Reduce from 1
```

### Model Loading Errors

**Problem**: Model fails to load or initialize

**Solution**:
1. Check model path in `models_config.yaml`
2. Verify model-specific environment is activated
3. Check external repository setup (see `external_repos/README.md`)

### Retry Logic Issues

**Problem**: Processing fails repeatedly

**Solution**: Check retry configuration in `models_config.yaml`:
```yaml
retry:
  max_attempts: 4  # Increase if needed
  frame_count_fallback: [256, 128, 64, 32]  # Adjust fallback values
```

### LLM Judge Errors

**Problem**: LLM judge evaluation fails

**Solution**:
```bash
# Skip LLM judge for faster evaluation
python 05_evaluation_inference/run_evaluation.py \
    --model gemini \
    --no-llm-judge
```

Or check LLM judge model configuration in `models_config.yaml`.

## Files

- `run_evaluation.py` - Main evaluation pipeline orchestrator
- `inference/run_inference.py` - Standalone inference script
- `metrics/compute_metrics.py` - Main metrics computation
- `metrics/t1_metrics.py` - Task 1 specific metrics
- `metrics/t2_metrics.py` - Task 2 specific metrics
- `metrics/t3_metrics.py` - Task 3 specific metrics
- `models_config.yaml` - Model and evaluation configuration
- `models/` - Model implementations (base_model, gemini, gpt4o, etc.)
- `prompts/` - Task-specific prompt templates
- `utils/` - Utility modules (audio_processor, frame_sampler, etc.)
- `models_requirements/` - Model-specific Python requirements
- `external_repos/` - External model repositories with fixes

## Notes

- **Environment Management**: Most models work with the general environment. Only use model-specific environments (in `models_requirements/`) when running models with special requirements (Uni-MoE, VITA, Video-LLaMA, MiniCPM-V, Phi-4).

- **Path Resolution**: All scripts expect to be run from the `sonic-o1` parent directory to properly resolve relative imports and data paths.

- **Resume Capability**: The pipeline automatically skips already processed entries. Use `--force-rerun` to reprocess everything.

- **Experiment Names**: Use `--experiment-name` to organize results for different experimental conditions (e.g., "frames_16", "modality_audio_only").

- **API Rate Limits**: API-based models (Gemini, GPT-4o) have rate limits. The pipeline includes automatic retry logic, but you may need to adjust delays for large-scale evaluation.

- **GPU Requirements**: Open-source models require GPUs. Ensure sufficient VRAM and CUDA setup before running GPU-based models.

- **External Dependencies**: Some models require external repositories with compatibility fixes (see `external_repos/README.md` for setup instructions).
