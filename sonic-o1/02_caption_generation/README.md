# WhisperX Caption Generation Setup

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
./configure --prefix=/projects/aixpert/users/ahmadradw/.local --enable-shared --disable-static --disable-x86asm
make -j4
make install

# Add to environment permanently
echo 'export PKG_CONFIG_PATH=/projects/aixpert/users/ahmadradw/.local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/projects/aixpert/users/ahmadradw/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
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
# Navigate to your project
cd /projects/aixpert/users/ahmadradw/VideoQA-Agentic

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

cd VideoAudioRepDataset
```

### Process All Topics
```bash
python process_whisper_captions.py --dataset-root .
```

### Process Specific Topics
```bash
python process_whisper_captions.py \
    --dataset-root . \
    --topics 01_Patient-Doctor_Consultations 02_Job_Interviews
```

### Choose Different Model Size
```bash
# Faster but less accurate
python process_whisper_captions.py --model base

# Most accurate (default)
python process_whisper_captions.py --model large-v2

# Even newer model
python process_whisper_captions.py --model large-v3
```

### Force English language (recommended)
```bash
# Add language='en' in the script to avoid misdetection
python process_whisper_captions.py --model large-v2
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

**Solution**: Add `language='en'` to transcribe call in script:
```python
result = model.transcribe(audio, batch_size=16, language='en')
```

### 6. CUDA Out of Memory

**Problem**: `CUDA out of memory`

**Solution**:
```bash
# Use smaller model
python process_whisper_captions.py --model base

# Or use int8 compute (modify script: compute_type="int8")
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

Compare generated captions with your existing ones:
```bash
# View existing caption
cat dataset/captions/01_Patient-Doctor_Consultations/caption_020.srt

# Test on single file
python whisper_test.py
```

## Environment Variables Summary

Add these to your `~/.bashrc` for permanent setup:
```bash
# FFmpeg
export PKG_CONFIG_PATH=/projects/aixpert/users/ahmadradw/.local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/projects/aixpert/users/ahmadradw/.local/lib:$LD_LIBRARY_PATH

# Cache directories (avoid disk quota)
export UV_CACHE_DIR=~/scratch/.uv_cache
export HF_HOME=~/scratch/.huggingface
export TORCH_HOME=~/scratch/.torch
export NLTK_DATA=~/scratch/nltk_data
export TMPDIR=~/scratch

# cuDNN
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import nvidia.cudnn 2>/dev/null && print(nvidia.cudnn.__path__[0])" 2>/dev/null)/lib
```
