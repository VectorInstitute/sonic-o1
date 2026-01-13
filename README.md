
# MAGNET Pipeline

Multi-agent framework for audio-visual question answering over large video collections.

## Repository Structure

After setup, your repository should look like this:

```
VideoQAReplication/
├── .venv/                          # Virtual environment (created by uv sync)
├── imagebind/                      # ImageBind repository (clone this)
│   ├── models/
│   ├── data/
│   └── ...
├── Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text/  # Dataset (download)
│   ├── videos/
│   │   ├── DIY/
│   │   ├── cooking-tutorials/
│   │   └── ...
│   ├── audio/
│   │   ├── DIY/
│   │   ├── cooking-tutorials/
│   │   └── ...
│   └── QA/
│       ├── DIY.json
│       ├── cooking-tutorials.json
│       └── ...
├── magnet/                         # Main package
│   ├── .env                        # API keys (create this, not tracked in git)
│   ├── pipeline.py
│   ├── av_rag.py
│   ├── sfs.py
│   ├── av_agent.py
│   ├── meta_agent.py
│   ├── models.py
│   └── load_data.py
├── scripts/                        # Test scripts
│   ├── test_av_rag.py
│   ├── test_av_agent.py
│   └── ...
├── results/                        # Output directory (created automatically)
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

**Important Notes:**
- `imagebind/` should be at the **project root** (same level as `magnet/`)
- `Videos-Dataset-For-LLMs-RAG.../` should also be at the **project root**
- `.env` file goes inside the `magnet/` directory
- `.venv/`, `imagebind/`, and the dataset are **not tracked** in git (see `.gitignore`)

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd VideoQAReplication
```

### 2. Environment with uv

```bash
uv sync
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
```

- `uv sync` creates `.venv/` (if it doesn't exist) AND installs all dependencies
- Just activate after `uv sync` completes

This installs all dependencies from `pyproject.toml`.

### 3. ImageBind Model

Clone the ImageBind repository into your project root:
```bash
# Make sure you're in the VideoQAReplication directory
git clone https://github.com/facebookresearch/ImageBind.git imagebind
```

**Link:** https://github.com/facebookresearch/ImageBind

**Note:** ImageBind dependencies are already included in `pyproject.toml`.

### 4. HuggingFace Cache

Set cache location to avoid filling home directory:
```bash
export HF_HOME=~/scratch/.cache/huggingface
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
mkdir -p ~/scratch/.cache/huggingface
```

Add to `~/.bashrc` to make permanent.

### 5. API Keys

Create `.env` file in the `magnet/` directory:
```bash
# Create the file
touch magnet/.env
```

Add your keys:
```
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

Get keys:
- **Gemini:** https://aistudio.google.com/app/apikey
- **OpenAI:** https://platform.openai.com/api-keys

### 6. Dataset

Download the dataset from HuggingFace and place it in the project root:

**Dataset Link:** https://huggingface.co/datasets/elmoghany/Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text

```bash
# After downloading, place it in the project root
# Your directory should be:
# VideoQAReplication/Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text/
```

#### Dataset Directory Structure

```
Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text/
├── videos/
│   ├── DIY/
│   │   ├── 0.mp4
│   │   ├── 1.mp4
│   │   ├── 2.mp4
│   │   └── ...
│   ├── cooking-tutorials/
│   │   ├── 0.mp4
│   │   ├── 1.mp4
│   │   └── ...
│   └── [other categories]/
├── audio/
│   ├── DIY/
│   │   ├── 0.m4a
│   │   ├── 1.m4a
│   │   ├── 2.m4a
│   │   └── ...
│   ├── cooking-tutorials/
│   │   ├── 0.m4a
│   │   ├── 1.m4a
│   │   └── ...
│   └── [other categories]/
└── QA/
    ├── DIY.json
    ├── cooking-tutorials.json
    └── [other categories].json
```

#### QA File Format Example

`QA/DIY.json`:
```json
[
  {
    "id": "1",
    "question": "How to do something?",
    "answer": "1) First step... 2) Second step...",
    "timestamps": {
      "0.txt": [[0, 36], [36, 63]],
      "1.txt": [[10, 45]]
    }
  },
  {
    "id": "2",
    "question": "What tools are needed?",
    "answer": "You will need...",
    "timestamps": {
      "2.txt": [[5, 30]],
      "5.txt": [[0, 20], [40, 60]]
    }
  }
]
```

**Note:** The `timestamps` dictionary maps video IDs (e.g., `"0.txt"` for `0.mp4`) to lists of `[start, end]` timestamp pairs in seconds.

## Usage

### Basic Run

```bash
cd magnet
python pipeline.py
```

Tests first 3 questions from DIY category. Results saved to `./results/diy_test_results.json`.

### With Screen (for long runs)

```bash
screen -S magnet
cd /path/to/VideoQAReplication/magnet
source ../.venv/bin/activate
export HF_HOME=~/scratch/.cache/huggingface
export TRANSFORMERS_CACHE=~/scratch/.cache/huggingface
python pipeline.py
```

Detach: `Ctrl+A` then `D`  
Reattach: `screen -r magnet`

## Pipeline Steps

1. **AV-RAG**: Retrieves top-k relevant videos using ImageBind embeddings
2. **SFS**: Selects salient frames from each video
3. **AV-Agents**: Processes each video with Qwen2.5-Omni (audio + visual)
4. **Meta-Agent**: Synthesizes final answer using GPT-4o

## Files

- `pipeline.py` - Main orchestration
- `av_rag.py` - Video retrieval using ImageBind + Gemini captions
- `sfs.py` - Salient frame selection algorithm
- `av_agent.py` - Individual video processing with Qwen2.5-Omni
- `meta_agent.py` - Response aggregation with GPT-4o
- `models.py` - ImageBind encoder and Gemini captioner
- `load_data.py` - Dataset loader

## Configuration

Edit `pipeline.py` main():

```python
DATASET_ROOT = "../Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text"
top_k = 6          # videos to retrieve
sfs_k = 8          # frames per video
sfs_m = 75         # uniform samples for SFS
sfs_gamma = 20.0   # temporal penalty
```

## Output

JSON file with:
- Question and ground truth
- Retrieved videos with scores
- Selected frame indices
- Temporal segments per video
- Final synthesized answer

## Memory Management

Pipeline clears GPU cache after each video and question to prevent OOM errors with multiple large models.

## Troubleshooting

**ImportError**: Run `uv sync` again  
**CUDA OOM**: Reduce `top_k` or `sfs_k`  
**API errors**: Check keys in `.env` inside `magnet/` directory  
**File not found**: Verify dataset and imagebind are in project root (not inside `magnet/`)  
**ImageBind errors**: Ensure ImageBind is cloned at project root: `VideoQAReplication/imagebind/`

## References

- **ImageBind:** https://github.com/facebookresearch/ImageBind
- **Dataset:** https://huggingface.co/datasets/elmoghany/Videos-Dataset-For-LLMs-RAG-That-Require-Audio-Vidoes-And-Text
- **Gemini API:** https://aistudio.google.com/app/apikey
- **OpenAI API:** https://platform.openai.com/api-keys
