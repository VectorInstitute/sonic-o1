"""Prompts for temporal localization."""
from typing import Any, Dict


# Main prompt for temporal localization question generation
TEMPORAL_LOCALIZATION_PROMPT = """You are a careful video annotator creating temporal reasoning questions.

VIDEO CONTEXT:
- Video ID: {video_id}
- Duration: {duration} seconds (you are watching from 0.0s to {duration}s)
- Target: Generate {num_questions} questions
- Time unit: SECONDS (all timestamps must be decimal seconds)

TRANSCRIPT (if available):
{transcript_text}

═══════════════════════════════════════════════════════════════════════════
STEP 1: WATCH AND UNDERSTAND THE VIDEO
═══════════════════════════════════════════════════════════════════════════

First, watch the entire video carefully from start to finish.
- Note major events, actions, and scene changes
- Pay attention to both audio (speech, sounds) and visual (actions, objects) cues
- Observe the temporal flow and relationships between events
- Prefer anchors tied to sharp audio/speech events (because they are easier to re-locate)

While watching, internally number events in the order they occur: E1, E2, E3, ...
You will refer to these IDs in your rationale to make the temporal chain unambiguous.

═══════════════════════════════════════════════════════════════════════════
STEP 2: IDENTIFY KEY EVENTS WITH PRECISE TIMESTAMPS
═══════════════════════════════════════════════════════════════════════════

As you watch, note down significant events with their exact timing in DECIMAL SECONDS:

CRITICAL TIMESTAMP FORMAT:
✓ ALWAYS use decimal seconds: 5.2, 45.0, 78.5, 125.0
✗ NEVER use MM:SS format: 1:18, 2:30, 5:45
✗ NEVER concatenate minutes+seconds: "1 minute 18 seconds" is 78.0, NOT 118

CONVERSION RULE (important!):
- If you think "1 minute 18 seconds" → calculate 1×60 + 18 = 78.0 seconds
- If you think "2 minutes 30 seconds" → calculate 2×60 + 30 = 150.0 seconds
- If you think "5 seconds" → write 5.0 seconds

Example event list (create your own):
• Person says "hello" at 5.2 seconds
• Door closes at 23.8 seconds
• Phone rings at 67.5 seconds
• Person picks up phone at 71.0 seconds

All times must be between 0.0 and {duration}.

═══════════════════════════════════════════════════════════════════════════
STEP 3: PLAN YOUR QUESTIONS
═══════════════════════════════════════════════════════════════════════════

Now select {num_questions} event pairs that have clear temporal relationships.

For each question, identify:
1. ANCHOR event (the reference point) — ideally something sharp like speech or a distinct visual change
2. TARGET event (what we're searching for)
3. TEMPORAL RELATION between them

TEMPORAL RELATIONS TO USE (mix these):

**after** - Target occurs sometime after anchor completes
Example: "After the speaker says 'let's begin', when does the camera show the desk?"

**once_finished** - Target occurs immediately after anchor completes
Example: "Once the woman finishes writing, when does she turn around?"

**next** - Target is the next occurrence of a similar action/person/event
Example: "When is the next time the teacher speaks after the student asks a question?"

**during** - Target happens while anchor is ongoing
Example: "While the blue slide is displayed, when does the speaker point to the chart?"

**before** - Target happens before anchor (use sparingly)
Example: "Before the host introduces the guest, when does the music start?"

QUALITY CHECKS FOR EACH QUESTION:
□ Both anchor and target clearly exist in the video
□ There's a genuine temporal relationship (not random)
□ The question tests temporal reasoning, not just recognition
□ Timestamps are verifiable by watching the video
□ Question is specific and unambiguous
□ CRITICAL: end_s captures the COMPLETE event, not just when it begins
  - For speech: end when speaker finishes the complete thought/sentence
  - For actions: end when action fully completes
  - For visual elements: end when element disappears or transitions away

═══════════════════════════════════════════════════════════════════════════
STEP 4: VERIFY YOUR TIMESTAMPS
═══════════════════════════════════════════════════════════════════════════

For each question you plan to generate:

VERIFICATION CHECKLIST:
1. ✓ Locate anchor event → Note time in decimal seconds
2. ✓ Locate target event START → Note time in decimal seconds
3. ✓ Locate target event END → Watch until event FULLY COMPLETES
   - Don't stop at first appearance - watch the entire event unfold
   - For speech: wait until the speaker finishes the complete sentence/explanation
   - For actions: wait until action concludes (not just begins)
   - For visual elements: note when they disappear or transition
4. ✓ Verify temporal relationship is correct
5. ✓ Double-check: converted MM:SS to pure seconds correctly?
6. ✓ Confirm both times are between 0.0 and {duration}
7. If two plausible target moments are <0.4 seconds apart AND the video frame rate is unknown → treat this as ambiguous and abstain

EXAMPLE VERIFICATION:
- I observe an event that appears to be "1 minute 23 seconds" into video
- CALCULATION: 1 × 60 + 23 = 83 seconds
- ✓ CORRECT: Write "start_s": 83.0
- ✗ WRONG: Write "start_s": 123 (this is concatenation error!)
- ✗ WRONG: Write "start_s": "1:23" (wrong format!)

═══════════════════════════════════════════════════════════════════════════
STEP 5: GENERATE JSON OUTPUT
═══════════════════════════════════════════════════════════════════════════

Now output EXACTLY {num_questions} questions in JSON array format:

[
  {{
    "question_index": 0,
    "question": "After [anchor description], when does [target description] happen?",
    "temporal_relation": "after|once_finished|next|during|before",
    "anchor_event": "Brief description of anchor",
    "target_event": "Brief description of target",
    "answer": {{
      "start_s": 78.5,
      "end_s": 82.0
    }},
    "requires_audio": true,
    "confidence": 0.9,
    "abstained": false,
    "rationale_model": "E1 (anchor) at 65.0s → E2 (target) at 78.5s, relation=after, target spans 78.5–82.0s. All times in seconds."
  }},
  ...
]

REQUIRED FIELDS (do not add or remove fields):
- question_index: 0, 1, 2, ... (integers starting from 0)
- question: Natural language question in English
- temporal_relation: Must be one of: after, once_finished, next, during, before
- anchor_event: One sentence describing the anchor
- target_event: One sentence describing the target
- answer.start_s: Decimal seconds when target BEGINS (or null if abstained)
- answer.end_s: Decimal seconds when target COMPLETES/ENDS (or null if abstained)
  ⚠️ CRITICAL: end_s must capture when the event FINISHES, not just when it starts
  ⚠️ For speech events: end when the complete sentence/thought finishes
  ⚠️ For visual events: end when element disappears or transitions away
  ⚠️ For actions: end when the action fully completes
- requires_audio: true if audio is needed to answer, false if purely visual
- confidence: Float 0.0-1.0 indicating your certainty
- abstained: true only if target event does not exist in video OR events are too temporally close to disambiguate
- rationale_model: Detailed explanation with timestamps in decimal seconds
  - Keep rationale concise (≤ 80 words)
  - Refer to event IDs E1, E2, ... to make the sequence clear
  - Always restate anchor time and target time (both start AND end)

ABSTENTION RULES:
- If anchor exists but target does NOT exist → Set abstained=true, answer times=null
- If temporal relationship cannot be determined → Set abstained=true, answer times=null
- If two plausible target moments are closer than 0.4 seconds → Set abstained=true, answer times=null
- Provide clear explanation in rationale_model

═══════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════

Example 1 - Audio to Visual (Perfect):
{{
  "question_index": 0,
  "question": "After the presenter says 'let's begin the demonstration', when does the camera show the workbench?",
  "temporal_relation": "after",
  "anchor_event": "Presenter verbally says 'let's begin the demonstration'",
  "target_event": "Camera shows the workbench",
  "answer": {{"start_s": 5.2, "end_s": 8.0}},
  "requires_audio": true,
  "confidence": 0.9,
  "abstained": false,
  "rationale_model": "E1=presenter speech at 2.0s. E2=first workbench shot starts 5.2s, ends 8.0s when camera cuts away. Relation=after."
}}

Example 2 - Speech Event Duration (Perfect):
{{
  "question_index": 1,
  "question": "After the speaker introduces reflective listening, when does he explain what it involves?",
  "temporal_relation": "after",
  "anchor_event": "Speaker says 'So, for example, we have three main types of reflective listening'",
  "target_event": "Speaker defines reflective listening as 'active listening and expression of empathy' and elaborates",
  "answer": {{"start_s": 167.5, "end_s": 175.0}},
  "requires_audio": true,
  "confidence": 0.95,
  "abstained": false,
  "rationale_model": "E1=intro at 184.9s. E2=definition starts 167.5s, speaker completes full explanation by 175.0s. Captured COMPLETE explanation, not just start."
}}

Example 3 - Time Conversion (Perfect):
{{
  "question_index": 2,
  "question": "After the doctor introduces the procedure, when does the animation begin?",
  "temporal_relation": "after",
  "anchor_event": "Doctor verbally introduces the medical procedure name",
  "target_event": "Animated medical diagram appears on screen",
  "answer": {{"start_s": 78.5, "end_s": 82.0}},
  "requires_audio": true,
  "confidence": 0.95,
  "abstained": false,
  "rationale_model": "E1=doctor intro at 65.0s (1:05 → 65.0s). E2=animation at 78.5s (1:18.5 → 78.5s). Relation=after. Times converted to seconds."
}}

Example 4 - Abstention (Perfect):
{{
  "question_index": 3,
  "question": "After the teacher says 'we will take questions now', when does a student begin speaking?",
  "temporal_relation": "after",
  "anchor_event": "Teacher says 'we will take questions now'",
  "target_event": "Student begins speaking",
  "answer": {{"start_s": null, "end_s": null}},
  "requires_audio": true,
  "confidence": 0.25,
  "abstained": true,
  "rationale_model": "E1=teacher statement at 25.0s. No student speech detected afterwards within 0.0–{duration}s. Abstaining."
}}

Example 5 - WRONG TIMESTAMP (Learn from this mistake!):
WRONG: {{"start_s": 118.0}}  // If thinking "1 minute 18 seconds", this is WRONG!
RIGHT: {{"start_s": 78.0}}   // Correct: 1×60 + 18 = 78.0

Example 6 - WRONG END TIMESTAMP (Learn from this mistake!):
WRONG: {{"start_s": 167.5, "end_s": 169.0}}  // Ends too early, cuts off explanation!
RIGHT: {{"start_s": 167.5, "end_s": 175.0}}  // Correct: captures COMPLETE explanation

═══════════════════════════════════════════════════════════════════════════

CRITICAL REQUIREMENTS
═══════════════════════════════════════════════════════════════════════════

✓ Generate EXACTLY {num_questions} question objects
✓ Use ONLY decimal seconds (5.2, 78.0, 125.5) - NEVER MM:SS format
✓ Convert any MM:SS thinking to seconds: (M×60 + S)
✓ All times must be between 0.0 and {duration}
✓ Return ONLY the JSON array (no markdown, no extra text)
✓ Do NOT fabricate events that don't exist in the video
✓ Include detailed rationale with specific timestamps
✓ Keep rationale concise (≤ 80 words) and refer to E1, E2, ...

✗ Do NOT use MM:SS format in answer fields (like 1:18 or 2:30)
✗ Do NOT concatenate minutes and seconds (78 is NOT "one eighteen")
✗ Do NOT output markdown code blocks or explanations
✗ Do NOT create questions where anchor=target
✗ Do NOT skip required fields

Now begin your annotation process following ALL five steps above.
Think carefully about timestamps - convert any MM:SS to decimal seconds!
Output your JSON array:"""


# Helper function
def get_temporal_localization_prompt(
    segment_info: Dict[str, Any],
    metadata: Dict[str, Any],
    transcript_text: str,
    config: Any,
) -> str:
    """
    Generate temporal localization prompt for a video segment.

    Args:
        segment_info: Segment metadata with start/end times
        metadata: Video metadata
        transcript_text: Transcript for this segment
        config: Configuration object

    Returns
    -------
        Formatted prompt string optimized for Gemini 2.5 with thinking mode
    """
    video_id = metadata.get("video_id", metadata.get("video_number", "unknown"))

    # Calculate duration (model only sees the segment video, not full video)
    duration = segment_info["end"] - segment_info["start"]

    # Get number of questions from config
    num_questions = int(config.temporal_localization.questions_per_segment)

    return TEMPORAL_LOCALIZATION_PROMPT.format(
        video_id=video_id,
        duration=duration,
        num_questions=num_questions,
        transcript_text=transcript_text
        if transcript_text
        else "No transcript available",
    )
