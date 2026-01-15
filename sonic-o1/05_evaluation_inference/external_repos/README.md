# External Repositories

This directory contains modified versions of external model repositories 

## Included Repositories

### Uni-MoE
- **Original:** https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs
- **License:** Apache 2.0 (see original repository)
- **Modifications:** Fixed import paths and dependencies for our evaluation pipeline
- **Credit:** Original authors from HITsz-TMG

### VideoLLaMA2
- **Original:** https://github.com/DAMO-NLP-SG/VideoLLaMA2
- **License:** Apache 2.0 (see LICENSE file)
- **Modifications:** Fixed video processing compatibility for our pipeline
- **Credit:** Original authors from DAMO-NLP-SG

### VITA
- **Original:** https://github.com/VITA-MLLM/VITA
- **License:** BSD 3-Clause (see License.txt)
- **Modifications:** Fixed audio processing and model loading for our pipeline
- **Credit:** Original VITA-MLLM team

## Usage
1. Clone the original repos
2. Some of those models have problems in distribute GPU inference, make sure to apply fixes

All original licenses and attributions are preserved in their respective directories.

## Attribution

We are grateful to the original authors of these repositories.
