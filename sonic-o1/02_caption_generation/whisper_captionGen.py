#!/usr/bin/env python3
"""whisper_captionGen.py.

Process audio files that need captions using WhisperX.

Generates YouTube-style SRT captions for videos without existing captions.

Author: SONIC-O1 Team
"""

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Dict, List

import torch
import whisperx
import yaml


def load_config(config_path: str = "config_whisper.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: List[Dict], max_chars_per_line: int = 42) -> str:
    """
    Convert WhisperX segments to SRT format.

    Similar to YouTube caption style with line breaking.
    """
    srt_lines = []

    for i, segment in enumerate(segments, 1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()

        # Format timestamps
        start_time = format_timestamp(start)
        end_time = format_timestamp(end)

        # Break long lines (YouTube typically uses ~42 chars per line)
        if len(text) > max_chars_per_line:
            words = text.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_chars_per_line and current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    current_line.append(word)
                    current_length += word_length

            if current_line:
                lines.append(" ".join(current_line))

            text = "\n".join(lines)

        # Add SRT entry
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries

    return "\n".join(srt_lines)


def transcribe_audio(audio_path: str, config: Dict) -> Dict:
    """
    Transcribe audio file using WhisperX with alignment.

    Args:
        audio_path: Path to audio file (.m4a)
        config: Configuration dictionary

    Returns
    -------
        Dictionary with aligned segments
    """
    model_cfg = config["model"]
    device = model_cfg["device"]
    language = model_cfg["language"]

    print(f"Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # 1. Transcribe with Whisper
    print(f"Loading Whisper model: {model_cfg['name']}")
    model = whisperx.load_model(
        model_cfg["name"],
        device,
        compute_type=model_cfg["compute_type"],
        language=language,  # Add language here at model load time
    )

    print(f"Transcribing (language: {language})...")

    # Transcribe - language already set in model
    result = model.transcribe(audio, batch_size=model_cfg["batch_size"])

    # Delete model to free memory
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # 2. Align whisper output for better timestamps
    print(f"Aligning transcription for language: {language}")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Delete alignment model
    del model_a
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return result


def process_topic(topic_path: Path, config: Dict):
    """
    Process all videos in a topic that need captions.

    Args:
        topic_path: Path to topic directory (captions/TOPIC_NAME)
        config: Configuration dictionary
    """
    needs_whisper_file = topic_path / "needs_whisper.txt"

    if not needs_whisper_file.exists():
        if config["processing"]["verbose"]:
            print(f"No needs_whisper.txt found in {topic_path.name}")
        return

    # Read audio files that need captions
    with open(needs_whisper_file, "r") as f:
        audio_files = [line.strip() for line in f if line.strip()]

    if not audio_files:
        if config["processing"]["verbose"]:
            print(f"No audio files need captions in {topic_path.name}")
        return

    print(f"\n{'=' * 60}")
    print(f"Processing topic: {topic_path.name}")
    print(f"Audio files to process: {len(audio_files)}")
    print(f"{'=' * 60}\n")

    # Update paths to match your structure
    dataset_root = topic_path.parent.parent  # Go up from captions/TOPIC to dataset/
    audios_dir = dataset_root / "audios" / topic_path.name
    captions_dir = topic_path  # Already in captions/TOPIC

    for audio_filename in audio_files:
        # Extract video ID from audio filename (e.g., audio_015.m4a -> 015)
        video_id = audio_filename.replace("audio_", "").replace(".m4a", "")

        audio_file = audios_dir / audio_filename
        caption_file = captions_dir / f"caption_{video_id}.srt"

        if not audio_file.exists():
            print(f"[WARNING] Audio file not found: {audio_file}")
            continue

        if caption_file.exists() and config["processing"]["skip_existing"]:
            if config["processing"]["verbose"]:
                print(f"[SKIP] Caption already exists: {caption_file}")
            continue

        print(f"\n[PROCESSING] Video: {video_id}")
        if config["processing"]["verbose"]:
            print(f"   Audio: {audio_file}")

        try:
            # Transcribe
            result = transcribe_audio(str(audio_file), config)

            # Convert to SRT
            srt_content = segments_to_srt(
                result["segments"],
                max_chars_per_line=config["output"]["max_chars_per_line"],
            )

            # Save SRT file
            with open(caption_file, "w", encoding="utf-8") as f:
                f.write(srt_content)

            print(f"[SUCCESS] Caption saved: {caption_file}")

            # Optionally save JSON with full details
            if config["output"]["save_json"]:
                json_file = captions_dir / f"caption_{video_id}.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                if config["processing"]["verbose"]:
                    print(f"   JSON saved: {json_file}")

        except Exception as e:
            print(f"[ERROR] Failed to process {video_id}: {e}")
            if config["processing"]["verbose"]:
                traceback.print_exc()


def main():
    """Run main processing function."""
    parser = argparse.ArgumentParser(
        description="Generate captions for videos using WhisperX"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_whisper.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    dataset_root = Path(config["dataset"]["root"])

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}")
        return

    # Get topics to process from captions directory
    topics = config["dataset"]["topics"]
    if topics:
        topic_dirs = [dataset_root / "captions" / topic for topic in topics]
        topic_dirs = [t for t in topic_dirs if t.exists()]
    else:
        # Process all topics
        captions_dir = dataset_root / "captions"
        topic_dirs = sorted([d for d in captions_dir.iterdir() if d.is_dir()])

    if not topic_dirs:
        print("[ERROR] No topics found to process")
        return

    print("\nStarting WhisperX caption generation")
    print(f"   Device: {config['model']['device']}")
    print(f"   Model: {config['model']['name']}")
    print(f"   Language: {config['model']['language'] or 'auto-detect'}")
    print(f"   Topics: {len(topic_dirs)}")

    # Process each topic
    for topic_dir in topic_dirs:
        try:
            process_topic(topic_dir, config)
        except Exception as e:
            print(f"[ERROR] Failed to process topic {topic_dir.name}: {e}")
            if config["processing"]["verbose"]:
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
