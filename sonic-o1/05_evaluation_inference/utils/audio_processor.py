"""
utils/audio_processor.py.

Shared audio chunking utilities for VITA and Qwen3.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import torch


logger = logging.getLogger(__name__)


def sample_audio_chunks(
    audio_features: torch.Tensor,
    audio_for_llm_lens: int,
    chunk_duration_sec: float,
    feature_rate: float,
    max_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Sample audio chunks uniformly from full audio.

    Args:
        audio_features: Full audio feature tensor (T, D)
        audio_for_llm_lens: Original LLM length
        chunk_duration_sec: Duration of each chunk in seconds
        feature_rate: Audio feature rate in Hz (12.5 for VITA, 25 for Qwen3)
        max_chunks: Maximum number of chunks to keep (None = keep all)

    Returns
    -------
        sampled_audio: Sampled audio features
        sampled_llm_lens: Adjusted LLM length
    """
    frames_per_chunk = int(chunk_duration_sec * feature_rate)
    total_frames = audio_features.shape[0]
    num_chunks = int(np.ceil(total_frames / frames_per_chunk))

    logger.info(
        f"Audio chunking: {total_frames} frames -> {num_chunks} chunks "
        f"of {chunk_duration_sec}s each"
    )

    # If max_chunks specified and we have more chunks, uniformly sample
    if max_chunks is not None and num_chunks > max_chunks:
        logger.info(
            f"Uniformly sampling {max_chunks} chunks from {num_chunks} total chunks"
        )

        # Create chunks
        chunks = []
        for i in range(num_chunks):
            start_idx = i * frames_per_chunk
            end_idx = min((i + 1) * frames_per_chunk, total_frames)
            chunks.append(audio_features[start_idx:end_idx])

        # Uniformly sample max_chunks from all chunks
        sample_indices = np.linspace(0, num_chunks - 1, max_chunks, dtype=int)
        sampled_chunks = [chunks[i] for i in sample_indices]

        # Concatenate sampled chunks
        sampled_audio = torch.cat(sampled_chunks, dim=0)

        # Adjust LLM length proportionally
        sampled_llm_lens = int(
            audio_for_llm_lens * (sampled_audio.shape[0] / total_frames)
        )

        logger.info(
            f"Final audio: {sampled_audio.shape[0]} frames (from {total_frames})"
        )
        return sampled_audio, sampled_llm_lens
    # Keep all chunks (no sampling needed)
    logger.info(f"Keeping all {num_chunks} chunks")
    return audio_features, audio_for_llm_lens
