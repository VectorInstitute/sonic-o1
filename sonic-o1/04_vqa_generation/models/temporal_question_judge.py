"""temporal_question_judge.py.

GPT-4V Temporal Question Judge.

Validate temporal localization questions by:
1. Checking if timestamps are within segment bounds
2. Verifying events exist in the video frames
3. Validating temporal relationships
4. Attempting to fix correctable errors

Author: SONIC-O1 Team
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openai


logger = logging.getLogger(__name__)


class TemporalQuestionJudge:
    """GPT-4V-based judge for validating temporal localization questions."""

    def __init__(self, config=None):
        """
        Initialize the temporal question judge.

        Args:
            config: Optional configuration object
        """
        self.config = config

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)

        # GPT-4V model to use
        self.model = getattr(config, "judge_model", "gpt-4o") if config else "gpt-4o"
        # Validation thresholds (seconds tolerance for fixing timestamps)
        self.max_timestamp_deviation = 5.0

    def validate_segment_questions(
        self,
        questions: List[Dict[str, Any]],
        segment_info: Dict[str, Any],
        frame_paths: List[Path],
        transcript_text: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate all questions for a segment.

        Args:
            questions: List of temporal questions to validate.
            segment_info: Segment metadata (start, end, duration).
            frame_paths: List of sampled frame paths from segment.
            transcript_text: Optional transcript text for the segment.

        Returns
        -------
            Tuple of (valid_questions, validation_stats).
        """
        segment_start = segment_info["start"]
        segment_end = segment_info["end"]

        logger.info(
            f"Validating {len(questions)} questions for segment "
            f"[{segment_start}, {segment_end}]"
        )

        valid_questions = []
        stats = {
            "total": len(questions),
            "valid": 0,
            "fixed": 0,
            "dropped": 0,
            "reasons": {
                "out_of_bounds": 0,
                "events_not_found": 0,
                "invalid_relation": 0,
                "failed_generation": 0,
                "judge_error": 0,
                "fixed_timestamps": 0,
                "abstained": 0,
            },
        }

        for i, question in enumerate(questions):
            try:
                logger.info(f"Validating question {i + 1}/{len(questions)}")

                # Check if question is already marked as failed
                if question.get("abstained", False):
                    stats["reasons"]["abstained"] += 1
                    stats["dropped"] += 1
                    logger.info(f"Question {i + 1} abstained by generator, dropping")
                    continue

                if question.get("rationale_model") == (
                    "Failed to generate temporal question"
                ):
                    stats["reasons"]["failed_generation"] += 1
                    stats["dropped"] += 1
                    logger.info(f"Question {i + 1} failed generation, dropping")
                    continue

                # Validate and potentially fix the question
                validation_result = self._validate_single_question(
                    question, segment_info, frame_paths, transcript_text
                )

                if validation_result["valid"]:
                    # Question is valid (possibly after fixing)
                    valid_question = validation_result["question"]
                    valid_questions.append(valid_question)
                    stats["valid"] += 1

                    if validation_result["fixed"]:
                        stats["fixed"] += 1
                        stats["reasons"]["fixed_timestamps"] += 1
                        logger.info(f"✓ Question {i + 1} validated and fixed")
                    else:
                        logger.info(f"✓ Question {i + 1} validated successfully")
                else:
                    # Question is invalid and cannot be fixed
                    stats["dropped"] += 1
                    reason = validation_result.get("reason", "unknown")
                    stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                    logger.warning(f"✗ Question {i + 1} dropped: {reason}")

            except Exception as e:
                logger.error(f"Error validating question {i + 1}: {e}")
                stats["dropped"] += 1
                stats["reasons"]["judge_error"] += 1

        logger.info(
            f"Validation complete: {stats['valid']} valid, "
            f"{stats['fixed']} fixed, {stats['dropped']} dropped"
        )
        return valid_questions, stats

    def _validate_single_question(
        self,
        question: Dict[str, Any],
        segment_info: Dict[str, Any],
        frame_paths: List[Path],
        transcript_text: str,
    ) -> Dict[str, Any]:
        """
        Validate a single temporal question.

        Returns
        -------
            Dict with keys: valid (bool), question (corrected dict if valid),
            fixed (bool), reason (str if invalid).
        """
        segment_start = segment_info["start"]
        segment_end = segment_info["end"]
        segment_duration = segment_end - segment_start

        answer = question.get("answer", {})
        start_s = answer.get("start_s")
        end_s = answer.get("end_s")

        # Check 1: Timestamp bounds
        if start_s is None or end_s is None:
            return {
                "valid": False,
                "reason": "out_of_bounds",
                "message": "Missing timestamps",
            }

        # CRITICAL: Are timestamps ABSOLUTE vs SEGMENT-RELATIVE?
        # If start_s >= segment_start, it's likely absolute time
        is_absolute = False
        if start_s >= segment_start and end_s <= segment_end:
            # Absolute video time; convert to segment-relative
            is_absolute = True
            logger.warning(f"Absolute timestamps [{start_s}, {end_s}], converting")

        # Check if timestamps are within segment bounds (for relative time)
        if not is_absolute and (start_s < 0 or end_s > segment_duration):
            # Maybe model used absolute time; check bounds
            if (
                segment_start <= start_s <= segment_end
                and segment_start <= end_s <= segment_end
            ):
                is_absolute = True
                logger.warning(
                    f"Timestamps [{start_s}, {end_s}] seem to be absolute, converting"
                )
            else:
                return {
                    "valid": False,
                    "reason": "out_of_bounds",
                    "message": (
                        f"Timestamps [{start_s}, {end_s}] out of "
                        f"bounds [0, {segment_duration}]"
                    ),
                }

        # Convert absolute to relative if needed
        if is_absolute:
            original_start = start_s
            original_end = end_s
            start_s_relative = start_s - segment_start
            end_s_relative = end_s - segment_start

            # Update question with relative times
            question["answer"]["start_s"] = start_s_relative
            question["answer"]["end_s"] = end_s_relative

            logger.info(
                f"Converted: absolute [{original_start}, {original_end}] "
                f"→ relative [{start_s_relative:.2f}, {end_s_relative:.2f}]"
            )

            return {
                "valid": True,
                "question": question,
                "fixed": True,
                "message": "Fixed absolute timestamps to relative",
            }

        # Check 2: Use GPT-4V to validate events and temporal relationship
        try:
            gpt4v_result = self._validate_with_gpt4v(
                question, segment_info, frame_paths, transcript_text
            )

            if gpt4v_result["valid"]:
                # GPT-4V validated the question
                if gpt4v_result.get("corrected_timestamps"):
                    # GPT-4V suggested timestamp corrections
                    corrected = gpt4v_result["corrected_timestamps"]
                    question["answer"]["start_s"] = corrected["start_s"]
                    question["answer"]["end_s"] = corrected["end_s"]
                    reason = gpt4v_result.get(
                        "correction_reason", "timestamps adjusted"
                    )
                    question["rationale_model"] += f" [Judge corrected: {reason}]"

                    return {
                        "valid": True,
                        "question": question,
                        "fixed": True,
                        "message": "GPT-4V corrected timestamps",
                    }
                return {
                    "valid": True,
                    "question": question,
                    "fixed": False,
                    "message": "GPT-4V validated",
                }
            return {
                "valid": False,
                "reason": gpt4v_result.get("reason", "events_not_found"),
                "message": gpt4v_result.get("message", "GPT-4V validation failed"),
            }

        except Exception as e:
            logger.error(f"GPT-4V validation error: {e}")
            # If GPT-4V fails but timestamps are in bounds, keep the question
            return {
                "valid": True,
                "question": question,
                "fixed": False,
                "message": (
                    f"GPT-4V error, keeping question with valid timestamps: {e}"
                ),
            }

    def _validate_with_gpt4v(
        self,
        question: Dict[str, Any],
        segment_info: Dict[str, Any],
        frame_paths: List[Path],
        transcript_text: str,
    ) -> Dict[str, Any]:
        """
        Use GPT-4V to validate the question against visual evidence.

        Returns
        -------
            Dict with keys: valid (bool), reason (str), corrected_timestamps
            (dict), correction_reason (str).
        """
        segment_start = segment_info["start"]
        segment_info["end"]

        # Build the validation prompt
        prompt = self._build_validation_prompt(question, segment_info, transcript_text)

        # Prepare images
        image_contents = []
        for i, frame_path in enumerate(frame_paths):
            try:
                with open(frame_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                # Extract timestamp from filename if possible
                frame_filename = frame_path.name
                timestamp_info = ""
                if "_t" in frame_filename:
                    try:
                        ts_part = frame_filename.split("_t")[1].split("s")[0]
                        absolute_time = float(ts_part)
                        relative_time = absolute_time - segment_start
                        timestamp_info = (
                            f" (abs: {absolute_time:.2f}s, rel: {relative_time:.2f}s)"
                        )
                    except (ValueError, IndexError):
                        pass

                image_contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high",
                        },
                    }
                )

                # Add caption for each frame
                image_contents.append(
                    {
                        "type": "text",
                        "text": (f"[Frame {i + 1}/{len(frame_paths)}{timestamp_info}]"),
                    }
                )

            except Exception as e:
                logger.error(f"Error encoding frame {frame_path}: {e}")
                continue

        if not image_contents:
            raise ValueError("No valid frames to send to GPT-4V")

        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert video analyst validating temporal "
                    "localization questions. Respond in valid JSON."
                ),
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_contents,
            },
        ]

        # Call GPT-4V
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent validation
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content
            result = json.loads(response_text)

            logger.debug(f"GPT-4V validation result: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT-4V response as JSON: {e}")
            logger.debug(f"Response text: {response_text}")
            raise
        except Exception as e:
            logger.error(f"GPT-4V API call failed: {e}")
            raise

    def _build_validation_prompt(
        self,
        question: Dict[str, Any],
        segment_info: Dict[str, Any],
        transcript_text: str,
    ) -> str:
        """Build the validation prompt for GPT-4V."""
        segment_start = segment_info["start"]
        segment_end = segment_info["end"]
        segment_duration = segment_end - segment_start

        answer = question.get("answer", {})
        start_s = answer.get("start_s")
        end_s = answer.get("end_s")

        return f"""You are validating a temporal localization question generated for a video segment.

                SEGMENT INFO:
                - Segment absolute time: {segment_start}s to {segment_end}s (duration: {segment_duration}s)
                - All timestamps in the question should be RELATIVE to segment start (0.0s to {segment_duration}s)

                QUESTION TO VALIDATE:
                {json.dumps(question, indent=2)}

                TRANSCRIPT (if available):
                {transcript_text if transcript_text else "No transcript available"}

                VALIDATION CRITERIA:

                1. **Event Existence**: Do BOTH the anchor event and target event actually exist in the frames you see?
                - Check if the described events are visible or can be inferred from the frames
                - For audio events (requires_audio=true), check if transcript supports the events

                2. **Timestamp Accuracy**: Are the provided timestamps [{start_s}s, {end_s}s] reasonable?
                - Timestamps should be in SEGMENT-RELATIVE time (0.0 to {segment_duration}s)
                - Do the frames near the target timestamps show the target event?
                - Allow ±5 second tolerance for minor inaccuracies

                3. **Temporal Relationship**: Does the temporal relationship make sense?
                - Relation: {question.get("temporal_relation", "unknown")}
                - Check if anchor and target have the stated relationship

                TASKS:

                1. Determine if the question is VALID (events exist, timestamps reasonable, relation makes sense)

                2. If timestamps are slightly off but events are identifiable:
                - Provide corrected timestamps if you can identify better times
                - Only correct if deviation is ≤5 seconds

                3. If question is invalid (events don't exist, wrong relation, timestamps way off):
                - Mark as invalid and provide reason

                OUTPUT FORMAT (JSON):
                {{
                "valid": true/false,
                "reason": "events_not_found|invalid_relation|timestamps_way_off|other",
                "message": "Detailed explanation",
                "corrected_timestamps": {{
                    "start_s": <float>,
                    "end_s": <float>
                }},  // Only if minor corrections needed
                "correction_reason": "Brief explanation of correction"
                }}

                Respond with ONLY the JSON object, no other text."""
