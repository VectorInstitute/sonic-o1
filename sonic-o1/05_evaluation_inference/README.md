# Evaluation & Inference Pipeline

This directory contains the evaluation and inference pipeline for video question-answering models. It supports multiple open-source and commercial models with proper environment management and metrics computation.

## Important Prerequisites

### 1. Working Directory
**IMPORTANT:** Always run scripts from the `sonic-o1` directory (parent directory), not from within `05_evaluation_inference`. This is required for relative paths to work correctly.

```bash
# Correct - run from sonic-o1 directory
cd /path/to/sonic-o1
python 05_evaluation_inference/run_evaluation.py --config configs/eval_config.yaml

# Incorrect - will fail due to broken relative paths
cd 05_evaluation_inference
python run_evaluation.py --config configs/eval_config.yaml
```

### 2. Environment Activation

**General Environment:** Most models use the general project environment defined in `../pyproject.toml` (parent directory of `sonic-o1`). Install and activate this environment first:

```bash
# From the parent directory containing pyproject.toml
pip install -e .
# or
pip install -r requirements.txt
```

**Model-Specific Environments:** Some models require specialized dependencies and have their own virtual environments. Only use these if you're running the specific models listed below:

Available model-specific environments (see `models_requirements/` directory):
- `venv_llama` - For Video-LLaMA models
- `venv_minicpm` - For MiniCPM-V models
- `venv_phi4` - For Phi-4 Vision models
- `venv_unimoe` - For Uni-MoE models
- `venv_vita` - For VITA models

Example activation for model-specific environments:
```bash
# Activate environment for Video-LLaMA
source venv_llama/bin/activate

# Activate environment for Uni-MoE
source venv_unimoe/bin/activate
```

**Rule of thumb:** Use the general environment unless the model has a `requirements_venv_<model>.txt` file in `models_requirements/`.

## Directory Structure

### Core Scripts
- [run_evaluation.py](run_evaluation.py) - Main evaluation pipeline orchestrator
- [inference/run_inference.py](inference/run_inference.py) - Standalone inference script for model predictions

### Configuration & Setup
- **configs/** - YAML configuration files for evaluation runs
- **models_requirements/** - Python requirements files for each model's virtual environment
  - `requirements_venv_llama.txt`
  - `requirements_venv_minicpm.txt`
  - `requirements_venv_phi4.txt`
  - `requirements_venv_unimoe.txt`
  - `requirements_venv_vita.txt`

### Model Implementations
- **models/** - Model wrapper classes (only `.py` files, no backup `.txt` files)
  - [base_model.py](models/base_model.py) - Base class for all models
  - [gemini.py](models/gemini.py) - Google Gemini API
  - [gpt4o.py](models/gpt4o.py) - OpenAI GPT-4o API
  - [minicpm.py](models/minicpm.py) - MiniCPM-V model
  - [phi4.py](models/phi4.py) - Phi-4 Vision model
  - [qwen3.py](models/qwen3.py) - Qwen3 VL model
  - [unimoe.py](models/unimoe.py) - Uni-MoE model
  - [videollama.py](models/videollama.py) - Video-LLaMA model
  - [vita.py](models/vita.py) - VITA model

### Metrics & Evaluation
- **metrics/** - Metric computation scripts
  - [compute_metrics.py](metrics/compute_metrics.py) - Main metrics computation
  - [t1_metrics.py](metrics/t1_metrics.py) - Task 1 specific metrics
  - [t2_metrics.py](metrics/t2_metrics.py) - Task 2 specific metrics
  - [t3_metrics.py](metrics/t3_metrics.py) - Task 3 specific metrics
  - [llm_judge_gpt.py](metrics/llm_judge_gpt.py) - GPT-based LLM judge
  - [llm_judge_qwen.py](metrics/llm_judge_qwen.py) - Qwen-based LLM judge

### Supporting Components
- **prompts/** - Task-specific prompt templates
  - [t1_prompts.py](prompts/t1_prompts.py)
  - [t2_prompts.py](prompts/t2_prompts.py)
  - [t3_prompts.py](prompts/t3_prompts.py)

- **utils/** - Utility functions for data processing
  - [audio_processor.py](utils/audio_processor.py) - Audio extraction and processing
  - [frame_sampler.py](utils/frame_sampler.py) - Video frame sampling strategies
  - [caption_handler.py](utils/caption_handler.py) - Caption/subtitle processing
  - [segmenter.py](utils/segmenter.py) - Video segmentation utilities
  - [config_loader.py](utils/config_loader.py) - Configuration management
  - [mm_process_pyav.py](utils/mm_process_pyav.py) - Multimedia processing with PyAV

- **external_repos/** - Open-source model repositories
  See [external_repos/README.md](external_repos/README.md) for details on included repositories (Uni-MoE, VideoLLaMA2, VITA) with compatibility fixes applied.

- **results/** - Output directory for evaluation results

## Usage Examples

### Running Full Evaluation
```bash
# From sonic-o1 directory, with correct env activated
cd /path/to/sonic-o1
source venv_llama/bin/activate  # Activate appropriate environment
python 05_evaluation_inference/run_evaluation.py \
    --config 05_evaluation_inference/configs/eval_config.yaml \
    --model videollama \
    --task t1
```

### Running Inference Only
```bash
# From sonic-o1 directory
cd /path/to/sonic-o1
source venv_unimoe/bin/activate
python 05_evaluation_inference/inference/run_inference.py \
    --model unimoe \
    --input_data data/test_videos.json \
    --output_dir 05_evaluation_inference/results/unimoe_inference
```

### Computing Metrics on Results
```bash
# From sonic-o1 directory
cd /path/to/sonic-o1
python 05_evaluation_inference/metrics/compute_metrics.py \
    --predictions 05_evaluation_inference/results/predictions.json \
    --ground_truth data/ground_truth.json \
    --task t1
```

## Common Workflow

1. **Setup Environment**
   ```bash
   cd /path/to/sonic-o1
   source venv_<model>/bin/activate  # Replace <model> with target model
   ```

2. **Run Inference**
   ```bash
   python 05_evaluation_inference/run_evaluation.py --config <config_file> --model <model_name>
   ```

3. **Compute Metrics**
   ```bash
   python 05_evaluation_inference/metrics/compute_metrics.py --predictions <pred_file> --ground_truth <gt_file>
   ```

4. **Review Results**
   - Check `results/` directory for output files
   - Prediction files contain model responses
   - Metric files contain computed evaluation scores

## Notes

- **Environment Management**: Most models work with the general environment defined in `../pyproject.toml`. Only use model-specific environments (in `models_requirements/`) when running models with special requirements (Uni-MoE, VITA, Video-LLaMA, MiniCPM-V, Phi-4). Mismatched environments will cause import errors or compatibility issues.

- **Path Resolution**: All scripts expect to be run from the `sonic-o1` parent directory to properly resolve relative imports and data paths.

- **Model Files**: The `models/` directory contains only final `.py` implementations. Backup `.txt` files are not included in the repository.

- **External Dependencies**: Some models require external repositories with fixes applied (see `external_repos/` directory).

- **API Models**: For Gemini and GPT-4o, ensure API keys are properly configured in your environment or `.env` file.
