# Caption Generation with WhisperX

This directory handles automatic caption generation for videos that don't have captions available from YouTube. It uses WhisperX for high-quality transcription with word-level timestamps.

## Prerequisites

Before running this step, you must have completed the data curation step (see [01_data_curation](../01_data_curation/)):
- Downloaded videos and audio files in `dataset/videos/` and `dataset/audios/`
- Generated `needs_whisper.txt` files listing audio files requiring transcription

## Installation

### 1. Build FFmpeg from source (if no sudo access)
```bash
# Download and extract FFmpeg
cd ~/scratch
wget https://ffmpeg.org/releases/ffmpeg-6.0.tar.xz
tar xf ffmpeg-6.0.tar.xz
cd ffmpeg-6.0

# Configure and build
export TMPDIR=~/scratch
./configure --prefix=../.local --enable-shared --disable-static --disable-x86asm
make -j4
make install

# Add to environment permanently
echo 'export PKG_CONFIG_PATH=../.local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=../.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Set cache directories to scratch (avoid disk quota issues)
```bash
# Set all cache directories to scratch
export UV_CACHE_DIR=~/scratch/.uv_cache
export HF_HOME=~/scratch/.huggingface
export TORCH_HOME=~/scratch/.torch
export NLTK_DATA=~/scratch/nltk_data

# Create directories
mkdir -p ~/scratch/.uv_cache ~/scratch/.huggingface ~/scratch/.torch ~/scratch/nltk_data

# Add to .bashrc permanently
echo 'export UV_CACHE_DIR=~/scratch/.uv_cache' >> ~/.bashrc
echo 'export HF_HOME=~/scratch/.huggingface' >> ~/.bashrc
echo 'export TORCH_HOME=~/scratch/.torch' >> ~/.bashrc
echo 'export NLTK_DATA=~/scratch/nltk_data' >> ~/.bashrc
```

### 3. Install WhisperX with uv pip (bypasses dependency issues)
```bash
# Navigate to working directory (note: sonic-o1/sonic-o1)
cd /path/to/sonic-o1/sonic-o1

# Activate environment
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
uv pip install --upgrade torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install WhisperX without dependencies first
uv pip install git+https://github.com/m-bain/whisperX.git --no-deps

# Install required dependencies
uv pip install faster-whisper pyannote-audio ctranslate2 onnxruntime nltk

# Install cuDNN for CUDA 12
uv pip install nvidia-cudnn-cu12

# Set cuDNN library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn 2>/dev/null && nvidia.cudnn.__path__[0]" 2>/dev/null)/lib' >> ~/.bashrc

# Download NLTK data
python << 'NLTK_EOF'
import nltk
import os
nltk_data_dir = os.path.expanduser('~/scratch/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
NLTK_EOF
```

### 4. Verify installation
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check WhisperX
python -c "import whisperx; print('WhisperX works')"

# Check cuDNN
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
```

## Configuration

Edit [config_whisper.yaml](config_whisper.yaml) to customize:
- Model settings (model size, device, language)
- Dataset paths
- Specific topics to process
- Output format options

### Key Configuration Options

```yaml
model:
  name: "large-v2"        # Model size: tiny, base, small, medium, large-v2, large-v3
  device: "cuda"          # cuda or cpu
  language: "en"          # Force English (recommended to avoid misdetection)

dataset:
  root: "dataset"         # Path to dataset directory from 01_data_curation
  topics: []              # Leave empty to process all, or specify topics
```

## Usage

### Request GPU node (SLURM) and ensure code running
```bash
# 1. Request GPU
srun --gres=gpu:1 --mem=32G --partition=a40 --pty bash

# 2. Activate environment
source .venv/bin/activate

# 3. Set environment variables (in case .bashrc didn't load)
export UV_CACHE_DIR=~/scratch/.uv_cache
export HF_HOME=~/scratch/.huggingface
export TORCH_HOME=~/scratch/.torch
export NLTK_DATA=~/scratch/nltk_data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib

# 4. Navigate to caption generation directory
cd /path/to/sonic-o1/sonic-o1/02_caption_generation
```

### Process All Topics
```bash
python whisper_captionGen.py
```

### Process Specific Topics
Edit [config_whisper.yaml](config_whisper.yaml) to specify topics:
```yaml
dataset:
  topics:
    - 01_Patient-Doctor_Consultations
    - 02_Job_Interviews
```
Then run:
```bash
python whisper_captionGen.py
```

### Use Different Configuration File
```bash
python whisper_captionGen.py --config my_config.yaml
```

## Model Size Comparison

| Model    | Parameters | Speed    | Accuracy | Use Case                    |
|----------|-----------|----------|----------|-----------------------------|
| tiny     | 39M       | ~32x     | Good     | Quick tests                 |
| base     | 74M       | ~16x     | Better   | Fast processing             |
| small    | 244M      | ~6x      | Good     | Balanced                    |
| medium   | 769M      | ~2x      | Better   | High quality                |
| large-v2 | 1550M     | 1x       | Best     | Production (recommended)    |
| large-v3 | 1550M     | 1x       | Best+    | Latest improvements         |

*Speed is relative to large-v2 on GPU*

## Output Format

The script generates:
- **SRT files**: `caption_XXX.srt` - YouTube-style captions
- **JSON files**: `caption_XXX.json` - Full transcription details with word-level timestamps

### SRT Format Example
```
1
00:00:04,720 --> 00:00:10,720
Hello folks I'm delighted today to be joined by 
Dr John Mckeown head of GP teaching and Dr Naomi

2
00:00:10,720 --> 00:00:15,720
Dow who is a GP and Senior clinical lecturer both 
from the University of Aberdeen
```

## Expected Processing Time

- **GPU (NVIDIA A40)**: 
  - ~0.5-2 minutes per video (with large-v2)
  - ~0.1-0.5 minutes per video (with base)

- **CPU**: 
  - ~5-15 minutes per video (with base)
  - Not recommended for large models

## Troubleshooting

### 1. Disk Quota Exceeded

**Problem**: `Disk quota exceeded (os error 122)`

**Solution**: Move all caches to scratch directory (see Installation step 2)
```bash
export UV_CACHE_DIR=~/scratch/.uv_cache
export HF_HOME=~/scratch/.huggingface
export TORCH_HOME=~/scratch/.torch
export NLTK_DATA=~/scratch/nltk_data
```

### 2. FFmpeg libraries not found

**Problem**: `Package libavformat was not found`

**Solution**: Build FFmpeg from source (see Installation step 1)

### 3. cuDNN library not found

**Problem**: `Unable to load libcudnn_cnn.so`

**Solution**:
```bash
uv pip install nvidia-cudnn-cu12
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0])")/lib
```

### 4. NLTK punkt_tab not found

**Problem**: `Resource punkt_tab not found`

**Solution**:
```bash
python -c "import nltk; nltk.download('punkt_tab', download_dir='~/scratch/nltk_data')"
```

### 5. Wrong language detected

**Problem**: Detects Welsh (cy) instead of English

**Solution**: Set language in [config_whisper.yaml](config_whisper.yaml):
```yaml
model:
  language: "en"
```

### 6. CUDA Out of Memory

**Problem**: `CUDA out of memory`

**Solution**: Use smaller model or int8 compute in [config_whisper.yaml](config_whisper.yaml):
```yaml
model:
  name: "base"              # Use smaller model
  compute_type: "int8"      # Use int8 for less memory
```

### 7. PyTorch version conflicts

**Problem**: WhisperX v3.7.4 requires PyTorch 2.8+

**Solution**: Install with `--no-deps` and manually install dependencies (see Installation step 3)

### 8. Slow Processing
```bash
# Check GPU is being used
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Quality Verification

After processing, verify the generated captions:
```bash
# View generated caption
cat dataset/captions/01_Patient-Doctor_Consultations/caption_001.srt

# Check how many captions were generated
ls dataset/captions/01_Patient-Doctor_Consultations/caption_*.srt | wc -l
```

The script automatically skips videos that already have captions (controlled by `skip_existing` in config).

## Environment Variables Summary

Add these to your `~/.bashrc` for permanent setup:
```bash
# FFmpeg
export PKG_CONFIG_PATH=../.local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=../.local/lib:$LD_LIBRARY_PATH

# Cache directories (avoid disk quota)
export UV_CACHE_DIR=~/scratch/.uv_cache
export HF_HOME=~/scratch/.huggingface
export TORCH_HOME=~/scratch/.torch
export NLTK_DATA=~/scratch/nltk_data
export TMPDIR=~/scratch

# cuDNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn 2>/dev/null && print(nvidia.cudnn.__path__[0])" 2>/dev/null)/lib
```
