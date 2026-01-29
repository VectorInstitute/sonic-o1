"""mcq_model.py.

Task 2: MCQ (Multiple Choice Questions) Generation Model.

Author: SONIC-O1 Team
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts.mcq_prompts import get_mcq_prompt
from utils.demographics_expander import DemographicsExpander
from utils.video_segmenter import VideoSegmenter

from .base_gemini import BaseGeminiClient


logger = logging.getLogger(__name__)


class MCQModel(BaseGeminiClient):
    """Generate segment-level MCQ VQA entries."""

    def __init__(self, config):
        """
        Initialize MCQ model.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.config = config
        self.segmenter = VideoSegmenter(config)
        self.demographics_expander = DemographicsExpander(config)

        # Get num_options from config for validation
        self.num_options = config.mcq.num_options
        self.option_letters = [chr(65 + i) for i in range(self.num_options)]

    def process_video(
        self,
        video_path: Path,
        audio_path: Optional[Path],
        transcript_path: Optional[Path],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Process a video and generate MCQ VQA entries (one per segment).

        Args:
            video_path: Path to video file.
            audio_path: Path to audio file (optional).
            transcript_path: Path to transcript/caption file (optional).
            metadata: Video metadata from metadata_enhanced.json.

        Returns
        -------
            List of VQA entry dicts for Task 2 (one per segment).
        """
        try:
            video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))
            duration = metadata.get("duration_seconds", 0)

            logger.info(f"Processing video {video_id} for MCQ (duration: {duration}s)")

            # Always segment videos (even short ones get 1 MCQ)
            video_segments = self.segmenter.segment_video(
                video_path, duration, task_type="mcq"
            )
            audio_segments = None
            if audio_path and audio_path.exists():
                audio_segments = self.segmenter.segment_audio(
                    audio_path, duration, task_type="mcq"
                )

            logger.info(f"Created {len(video_segments)} segments for MCQ generation")

            # Generate MCQ for each segment
            mcq_entries = []
            for i, seg in enumerate(video_segments):
                try:
                    # Get corresponding audio segment
                    audio_seg_path = (
                        audio_segments[i]["segment_path"] if audio_segments else None
                    )

                    # Extract transcript for this segment
                    transcript_text = ""
                    if transcript_path and transcript_path.exists():
                        transcript_text = self.segmenter.extract_transcript_segment(
                            transcript_path,
                            seg["start"],
                            seg["end"],
                            strip_timestamps=True,
                        )

                    # Generate MCQ for this segment
                    mcq_entry = self._generate_mcq_for_segment(
                        seg, audio_seg_path, transcript_text, metadata
                    )

                    if mcq_entry:
                        mcq_entries.append(mcq_entry)

                except Exception as e:
                    logger.error(f"Failed to generate MCQ for segment {i}: {e}")
                    continue

            # Cleanup temporary segment files
            try:
                self.segmenter.cleanup_segments(video_segments)
                if audio_segments:
                    self.segmenter.cleanup_segments(audio_segments)
            except Exception as e:
                logger.warning(f"Failed to cleanup segments: {e}")

            logger.info(
                f"Generated {len(mcq_entries)} MCQ entries for video {video_id}"
            )
            return mcq_entries

        except Exception as e:
            logger.error(
                f"Error processing video {video_id} for MCQ: {e}", exc_info=True
            )
            return []

    def _generate_mcq_for_segment(
        self,
        segment_info: Dict,
        audio_path: Optional[Path],
        transcript_text: str,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Generate one MCQ for a video segment with retry on parse failure."""
        video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))
        seg_num = segment_info["segment_number"]

        logger.info(
            f"Generating MCQ for {video_id} segment {seg_num} "
            f"({segment_info['start']}s-{segment_info['end']}s)"
        )

        max_attempts = 3
        mcq_data = None

        # Prepare media files once (outside retry loop)
        media_files = []
        seg_path = segment_info["segment_path"]
        if seg_path.exists():
            # Determine if it's video or audio
            if seg_path.suffix.lower() in [
                ".mp4",
                ".avi",
                ".mov",
                ".webm",
                ".mkv",
                ".m4v",
            ]:
                media_files.append(("video", seg_path))
            else:
                media_files.append(("audio", seg_path))

        if audio_path and audio_path.exists():
            media_files.append(("audio", audio_path))

        # Retry loop for MCQ generation
        for attempt in range(max_attempts):
            try:
                # Build MCQ generation prompt
                prompt = get_mcq_prompt(
                    segment_info, metadata, transcript_text, self.config
                )

                # On retry, add explicit JSON validation instructions
                if attempt > 0:
                    prompt += (
                        "\n\nPREVIOUS ATTEMPT RETURNED INVALID JSON. "
                        "Critical requirements:\n"
                    )
                    prompt += f"- Return ONLY valid JSON with exactly {self.num_options} options\n"
                    prompt += (
                        "- Use commas between all properties/elements except the last\n"
                    )
                    prompt += f"- Last option MUST be '({self.option_letters[-1]}) Not enough evidence'\n"
                    prompt += (
                        f"- Include 'answer_index' (0-{self.num_options - 1}) "
                        f"and 'answer_letter' (A-{self.option_letters[-1]})\n"
                    )
                    prompt += "- Use double quotes for all strings\n"
                    prompt += "- No trailing commas before closing brackets\n"

                # Generate MCQ
                response_text = self.generate_content(
                    media_files, prompt, video_fps=0.5
                )
                mcq_data = self._parse_mcq_response(response_text)

                # Check if parsing succeeded (confidence > 0 and rationale valid)
                if (
                    mcq_data.get("confidence", 0) > 0
                    and mcq_data.get("rationale", "") != "Failed to generate MCQ"
                ):
                    break
                if attempt < max_attempts - 1:
                    time.sleep(30)

            except Exception as e:
                logger.error(
                    f"Failed to generate MCQ for segment {seg_num}: {e}", exc_info=True
                )
                if attempt < max_attempts - 1:
                    time.sleep(30)

        # If all attempts failed, use default MCQ
        if (
            not mcq_data
            or mcq_data.get("confidence", 0) == 0
            or mcq_data.get("rationale", "") == "Failed to generate MCQ"
        ):
            mcq_data = self._get_default_mcq()

        # Get demographics for this segment
        try:
            demographics_data = self._get_segment_demographics(
                segment_info, seg_path, audio_path, transcript_text, metadata
            )
        except Exception as e:
            demographics_data = {
                "demographics": [],
                "total_individuals": 0,
                "confidence": 0.0,
                "explanation": f"Error generating demographics: {str(e)}",
            }

        # Build MCQ entry with all demographic information
        return {
            "video_id": video_id,
            "video_number": metadata.get("video_number", video_id),
            "segment": {"start": segment_info["start"], "end": segment_info["end"]},
            "question": mcq_data.get("question", ""),
            "options": mcq_data.get("options", []),
            "answer_index": mcq_data.get("answer_index", self.num_options - 1),
            "answer_letter": mcq_data.get("answer_letter", self.option_letters[-1]),
            "rationale": mcq_data.get("rationale", ""),
            "evidence_tags": mcq_data.get("evidence_tags", []),
            "requires_audio": mcq_data.get("requires_audio", False),
            "demographics": demographics_data.get("demographics", []),
            "demographics_total_individuals": demographics_data.get(
                "total_individuals", 0
            ),
            "demographics_confidence": demographics_data.get("confidence", 0.0),
            "demographics_explanation": demographics_data.get("explanation", ""),
            "confidence": mcq_data.get("confidence", 0.0),
        }

    def _get_segment_demographics(
        self,
        segment_info: Dict,
        video_path: Path,
        audio_path: Optional[Path],
        transcript_text: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get expanded demographics for a specific segment."""
        try:
            # Get human-reviewed demographics from metadata (video-level)
            human_demographics = metadata.get("demographics_detailed_reviewed", {})
            if not human_demographics:
                logger.warning(
                    f"No human-reviewed demographics for {metadata.get('video_id')}"
                )
                return {
                    "demographics": [],
                    "total_individuals": 0,
                    "confidence": 0.0,
                    "explanation": "No human-reviewed demographics available",
                }

            # Build expansion prompt (segment-level)
            prompt = self.demographics_expander.build_expansion_prompt(
                human_demographics,
                segment_info={
                    "start": segment_info["start"],
                    "end": segment_info["end"],
                },
            )

            # Prepare media files
            media_files = []
            if video_path and video_path.exists():
                if video_path.suffix.lower() in [
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".webm",
                    ".mkv",
                    ".m4v",
                ]:
                    media_files.append(("video", video_path))
                else:
                    media_files.append(("audio", video_path))

            if audio_path and audio_path.exists():
                media_files.append(("audio", audio_path))

            # Add transcript context to prompt
            if transcript_text:
                prompt += f"\n\nSEGMENT TRANSCRIPT:\n{transcript_text[:1000]}"

            # Generate demographics
            response_text = self.generate_content(media_files, prompt, video_fps=0.25)
            demographics_data = self.demographics_expander.parse_demographics_response(
                response_text
            )

            # Ensure all fields are present
            if "explanation" not in demographics_data:
                demographics_data["explanation"] = "No explanation provided"

            return demographics_data

        except Exception as e:
            logger.error(f"Failed to get segment demographics: {e}")
            return {
                "demographics": [],
                "total_individuals": 0,
                "confidence": 0.0,
                "explanation": f"Error generating demographics: {str(e)}",
            }

    def _extract_json_from_markdown(self, text: str) -> str:
        """Extract JSON from markdown code blocks."""
        text = text.strip()
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.rfind("```")
            if end > start:
                return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.rfind("```")
            if end > start:
                return text[start:end].strip()
        return text

    def _normalize_mcq_data(self, data: Any) -> Optional[Dict[str, Any]]:
        """Normalize parsed data to dict format."""
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                logger.warning("API returned list, extracting first element")
                return data[0]
            logger.error("API returned invalid list format")
            return None

        if not isinstance(data, dict):
            logger.error(f"Parsed data is not a dict, got {type(data)}")
            return None

        return data

    def _normalize_options_count(self, options: List[str]) -> List[str]:
        """Normalize options count to expected num_options."""
        expected_num_options = self.num_options

        if len(options) != expected_num_options:
            logger.warning(
                f"MCQ has {len(options)} options, expected {expected_num_options}"
            )
            # Pad with "Not enough evidence" if needed
            while len(options) < expected_num_options:
                last_letter = self.option_letters[len(options)]
                options.append(f"({last_letter}) Not enough evidence")
            options = options[:expected_num_options]

        # Ensure last option is "Not enough evidence"
        last_letter = self.option_letters[-1]
        if "Not enough evidence" not in options[-1].lower():
            options[-1] = f"({last_letter}) Not enough evidence"

        return options

    def _format_option_with_letter(self, option: str, letter: str) -> str:
        """Format option with letter prefix, removing any existing prefix."""
        cleaned = option.strip()
        # Remove any existing letter prefix
        for existing_letter in self.option_letters:
            if cleaned.startswith(f"({existing_letter})"):
                cleaned = cleaned[3:].strip()
                break
        return f"({letter}) {cleaned}"

    def _format_options_with_letters(self, options: List[str]) -> List[str]:
        """Format all options with correct letter prefixes."""
        formatted = []
        for i, option in enumerate(options):
            letter = self.option_letters[i]
            formatted.append(self._format_option_with_letter(option, letter))
        return formatted

    def _validate_answer_fields(
        self, data: Dict[str, Any]
    ) -> tuple[int, str]:
        """Validate and normalize answer_index and answer_letter."""
        expected_num_options = self.num_options
        answer_index = int(data.get("answer_index", expected_num_options - 1))
        max_index = expected_num_options - 1

        if answer_index < 0 or answer_index > max_index:
            logger.warning(
                f"Invalid answer_index {answer_index}, defaulting to {max_index}"
            )
            answer_index = max_index

        answer_letter = data.get("answer_letter", self.option_letters[answer_index])
        expected_letter = self.option_letters[answer_index]

        if answer_letter != expected_letter:
            logger.warning(
                f"Mismatch answer_letter={answer_letter}, "
                f"answer_index={answer_index} (expected {expected_letter})"
            )
            answer_letter = expected_letter

        return answer_index, answer_letter

    def _parse_mcq_response(self, response_text: str) -> Dict[str, Any]:
        """Parse MCQ response from Gemini."""
        try:
            cleaned_text = self._extract_json_from_markdown(response_text)
            data = json.loads(cleaned_text)

            normalized_data = self._normalize_mcq_data(data)
            if normalized_data is None:
                return self._get_default_mcq()

            # Normalize options count and format
            options = normalized_data.get("options", [])
            options = self._normalize_options_count(options)
            formatted_options = self._format_options_with_letters(options)

            # Validate answer fields
            answer_index, answer_letter = self._validate_answer_fields(normalized_data)

            return {
                "question": normalized_data.get(
                    "question", "What is happening in the video and audio?"
                ),
                "options": formatted_options,
                "answer_index": answer_index,
                "answer_letter": answer_letter,
                "rationale": normalized_data.get("rationale", ""),
                "evidence_tags": normalized_data.get("evidence_tags", []),
                "requires_audio": bool(normalized_data.get("requires_audio", False)),
                "confidence": float(normalized_data.get("confidence", 0.0)),
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCQ JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            return self._get_default_mcq()
        except Exception as e:
            logger.error(f"Error parsing MCQ response: {e}")
            logger.debug(
                f"Response (500 chars): "
                f"{response_text[:500] if response_text else 'None'}..."
            )
            return self._get_default_mcq()

    def _get_default_mcq(self) -> Dict[str, Any]:
        """Return default MCQ structure when parsing fails."""
        # Build default options dynamically based on num_options
        default_options = []
        for i in range(self.num_options - 1):
            letter = self.option_letters[i]
            default_options.append(f"({letter}) Unable to generate option")

        # Last option is always "Not enough evidence"
        last_letter = self.option_letters[-1]
        default_options.append(f"({last_letter}) Not enough evidence")

        return {
            "question": "What is happening in the video and audio?",
            "options": default_options,
            "answer_index": self.num_options - 1,
            "answer_letter": last_letter,
            "rationale": "Failed to generate MCQ",
            "evidence_tags": [],
            "requires_audio": False,
            "confidence": 0.0,
        }
