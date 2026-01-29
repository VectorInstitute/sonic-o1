"""Prompts for GPT-4V Temporal Question Validation."""

import json
from typing import Any, Dict


def build_validation_prompt(
    question: Dict[str, Any], segment_info: Dict[str, Any], transcript_text: str = ""
) -> str:
    """
    Build GPT-4V validation prompt for a temporal question.

    Args:
        question: The temporal question to validate
        segment_info: Segment metadata (start, end)
        transcript_text: Optional transcript text

    Returns
    -------
        Formatted validation prompt
    """
    segment_start = segment_info["start"]
    segment_end = segment_info["end"]
    segment_duration = segment_end - segment_start

    answer = question.get("answer", {})
    start_s = answer.get("start_s")
    end_s = answer.get("end_s")

    return f"""You are validating a temporal localization question for a video segment.

═══════════════════════════════════════════════════════════════════════════
SEGMENT INFORMATION
═══════════════════════════════════════════════════════════════════════════

- Segment absolute time: {segment_start}s to {segment_end}s
- Segment duration: {segment_duration}s
- All timestamps in the question MUST be RELATIVE to segment start (0.0s to {segment_duration}s)

═══════════════════════════════════════════════════════════════════════════
QUESTION TO VALIDATE
═══════════════════════════════════════════════════════════════════════════

{json.dumps(question, indent=2)}

═══════════════════════════════════════════════════════════════════════════
TRANSCRIPT (if available)
═══════════════════════════════════════════════════════════════════════════

{transcript_text if transcript_text else "No transcript available for this segment."}

═══════════════════════════════════════════════════════════════════════════
VALIDATION CRITERIA
═══════════════════════════════════════════════════════════════════════════

1. **EVENT EXISTENCE**
   - Does the ANCHOR event actually exist in the frames you see?
   - Does the TARGET event actually exist in the frames you see?
   - For audio-required events (requires_audio=true), check if transcript supports them
   - If either event does NOT exist → mark as INVALID

2. **TIMESTAMP ACCURACY**
   - Provided timestamps: [{start_s}s, {end_s}s]
   - Are these timestamps in SEGMENT-RELATIVE time (0.0 to {segment_duration}s)?
   - Does start_s match when the target event BEGINS?
   - Does end_s match when the target event FULLY COMPLETES?
     * CRITICAL: Check if end_s is too early (event continues after end_s)
     * For speech: end_s should be after the complete sentence/thought finishes
     * For actions: end_s should be after action completes
     * For visual elements: end_s should be when element disappears
   - Allow ±5 second tolerance for minor inaccuracies
   - If end_s cuts off the event prematurely → provide correction

3. **TEMPORAL RELATIONSHIP**
   - Stated relation: {question.get("temporal_relation", "unknown")}
   - Does this relationship make sense between anchor and target?
   - Relations:
     * "after" = target occurs sometime after anchor completes
     * "once_finished" = target occurs immediately after anchor
     * "next" = target is next occurrence of similar event
     * "during" = target happens while anchor is ongoing
     * "before" = target happens before anchor

═══════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════

1. **DETERMINE VALIDITY**
   - If events exist AND timestamps reasonable AND relation correct → VALID
   - If events don't exist OR timestamps way off OR relation wrong → INVALID

2. **MINOR CORRECTIONS (if applicable)**
   - If timestamps are slightly off (≤5 seconds) but you can clearly identify when events occur:
     * Provide corrected timestamps
     * Only correct if deviation is within ±5 seconds
   - If timestamps are off by >5 seconds → mark INVALID

3. **INVALIDATION REASONS**
   - "events_not_found" - Anchor or target doesn't exist in video
   - "invalid_relation" - Temporal relationship doesn't match reality
   - "timestamps_way_off" - Timestamps off by >5 seconds and cannot be corrected
   - "other" - Other validation failure

═══════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON ONLY)
═══════════════════════════════════════════════════════════════════════════

Respond with ONLY a JSON object (no other text):

{{
  "valid": true/false,
  "reason": "events_not_found|invalid_relation|timestamps_way_off|other",
  "message": "Detailed explanation of your validation decision",
  "corrected_timestamps": {{
    "start_s": <float>,
    "end_s": <float>
  }},  // ONLY include if making minor corrections (≤5s deviation)
  "correction_reason": "Brief explanation of why you corrected timestamps"
}}

EXAMPLES:

Example 1 - Valid question, no correction needed:
{{
  "valid": true,
  "message": "Both anchor (speaker saying 'let's begin') and target (camera shows desk) are clearly visible in frames. Timestamps align with observed events. Temporal relation 'after' is correct."
}}

Example 2 - Valid question, minor timestamp correction:
{{
  "valid": true,
  "message": "Events exist and relation is correct, but target event appears 3 seconds earlier than stated timestamp.",
  "corrected_timestamps": {{
    "start_s": 12.5,
    "end_s": 15.0
  }},
  "correction_reason": "Target event (door closing) occurs at 12.5s, not 15.5s as originally stated"
}}

Example 2b - Valid question, end timestamp too short:
{{
  "valid": true,
  "message": "Start timestamp is accurate, but end timestamp cuts off before event completes. Speaker continues explaining until 15.8s.",
  "corrected_timestamps": {{
    "start_s": 7.5,
    "end_s": 15.8
  }},
  "correction_reason": "Original end_s of 11.2s was premature. Speaker finishes complete explanation at 15.8s."
}}

Example 3 - Invalid question, events don't exist:
{{
  "valid": false,
  "reason": "events_not_found",
  "message": "Anchor event (person saying 'hello') is not observed in any frame or supported by transcript. Cannot validate this question."
}}

Example 4 - Invalid question, wrong temporal relation:
{{
  "valid": false,
  "reason": "invalid_relation",
  "message": "Target event (animation appearing) occurs BEFORE anchor event (speaker introduction), but question claims relation is 'after'. Temporal order is reversed."
}}

═══════════════════════════════════════════════════════════════════════════

Begin your validation. Respond with ONLY the JSON object."""


def build_batch_validation_system_prompt() -> str:
    """
    System prompt for GPT-4V judge validation.

    Returns
    -------
        System prompt string
    """
    return """You are an expert video analyst specializing in temporal event validation.

Your role is to:
1. Carefully examine video frames to identify events
2. Validate temporal relationships between events
3. Verify timestamp accuracy
4. Provide corrections only when confident and deviation is small (≤5 seconds)

You must respond ONLY with valid JSON. Never include explanations outside the JSON structure.

Key principles:
- Be strict about event existence - if you don't see it, mark as invalid
- Allow minor timestamp adjustments (≤5s) if events are clearly identifiable
- Validate temporal relationships carefully (after, before, during, etc.)
- Use transcript to validate audio events when available
- When uncertain, mark as invalid rather than guessing"""
