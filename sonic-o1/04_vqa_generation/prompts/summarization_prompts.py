"""Task 1: Summarization prompts (map and reduce phases)."""

import json

# MAP PHASE: Per-segment summarization
MAP_PHASE_PROMPT = """You are a precise video segment summarizer.

SEGMENT INFORMATION:
- Segment time: {start_time}s to {end_time}s ({duration}s duration)
- Video title: {title}
- Topic: {topic_name}

TRANSCRIPT/CAPTIONS:
{transcript_text}

YOUR TASK:
Summarize this video segment concisely and accurately.

RULES:
- Maximum {max_words} words
- Keep strict chronology - describe events in the order they occur
- Prefer facts that are audible in the transcript or visible in the video
- If nothing meaningful happens, state "No salient events in this segment"
- Be specific: include names, actions, objects, and key details
- Do NOT speculate beyond what you see/hear

OUTPUT FORMAT (JSON):
{{
  "segment_start": "{start_time}",
  "segment_end": "{end_time}",
  "summary": "120 words maximum describing what happens in this segment...",
  "mini_timeline": [
    {{"time": "MM:SS", "title": "Event name", "note": "One line description"}},
    {{"time": "MM:SS", "title": "Another event", "note": "Brief detail"}}
  ],
  "entities": ["names", "objects", "places", "key terms mentioned"],
  "confidence": 0.85
}}

CRITICAL:
- Return ONLY valid JSON
- No markdown, no extra text
- Timestamps in mini_timeline should be relative to SEGMENT start time
- Include 2-5 timeline items for this segment

Begin analysis:"""


# REDUCE PHASE: Merge segments into video-level summary
REDUCE_PHASE_PROMPT = """You are an editor merging ordered segment summaries into a comprehensive video summary.

VIDEO INFORMATION:
- Video ID: {video_id}
- Title: {title}
- Topic: {topic_name}
- Total duration: {duration}s
- Number of segments: {num_segments}

SEGMENT SUMMARIES:
{segment_summaries_json}

YOUR TASK:
Merge these segment summaries into a single comprehensive video summary.

PRODUCE:
1. TL;DR: {num_bullets} concise bullet points
2. Detailed summary: {max_words_detailed} words (structure: Purpose → Key Points → Outcomes)
3. Global timeline: {timeline_min}-{timeline_max} items covering the entire video
4. Glossary: {glossary_min}-{glossary_max} key entities/terms from the full video

RULES:
- Remove duplicates across segments
- Keep strict chronological order
- Be concise and factual
- Prefer events that appear in multiple segments or are clearly important
- Timeline should span from video start to end with actual timestamps (HH:MM:SS format)
- Glossary should include: people's names, important objects, locations, technical terms

OUTPUT FORMAT (JSON):
{{
  "summary_short": [
    "• First key point...",
    "• Second key point...",
    "• Third key point...",
    "• Fourth key point...",
    "• Fifth key point..."
  ],
  "summary_detailed": "200-300 word comprehensive summary. Start with the video's purpose, cover key points in chronological order, and conclude with outcomes or main takeaways...",
  "timeline": [
    {{"start": "00:00:12", "end": "00:00:45", "title": "Introduction", "note": "Brief description"}},
    {{"start": "00:05:30", "end": "00:06:15", "title": "Main topic", "note": "Key details"}},
    {{"start": "00:12:00", "end": "00:13:30", "title": "Conclusion", "note": "Final points"}}
  ],
  "glossary": ["Entity 1", "Entity 2", "Important Term", "Key Person", "Location"],
  "confidence": 0.88
}}

CRITICAL:
- Return ONLY valid JSON
- No markdown, no extra text
- Timeline timestamps must be in HH:MM:SS format relative to VIDEO start
- Confidence should reflect how well segments agreed and information quality

Begin merging:"""


# DIRECT SUMMARIZATION: For short videos (no segmentation needed)
DIRECT_SUMMARY_PROMPT = """You are a precise video summarizer.

VIDEO INFORMATION:
- Video ID: {video_id}
- Title: {title}
- Topic: {topic_name}
- Duration: {duration}s

TRANSCRIPT/CAPTIONS:
{transcript_text}

YOUR TASK:
Create a comprehensive summary of this video.

PRODUCE:
1. TL;DR: {num_bullets} concise bullet points
2. Detailed summary: {max_words_detailed} words (structure: Purpose → Key Points → Outcomes)
3. Timeline: {timeline_min}-{timeline_max} key moments with timestamps
4. Glossary: {glossary_min}-{glossary_max} important entities/terms

RULES:
- Keep strict chronological order
- Be factual - only include what you see/hear
- No speculation
- Timeline should cover the full video with actual timestamps (HH:MM:SS)
- Glossary should include: people's names, objects, locations, key terms

OUTPUT FORMAT (JSON):
{{
  "summary_short": [
    "• First key point...",
    "• Second key point...",
    "• Third key point...",
    "• Fourth key point...",
    "• Fifth key point..."
  ],
  "summary_detailed": "200-300 word comprehensive summary covering the entire video...",
  "timeline": [
    {{"start": "00:00:10", "end": "00:00:35", "title": "Intro", "note": "What happens"}},
    {{"start": "00:01:15", "end": "00:01:45", "title": "Main point", "note": "Details"}}
  ],
  "glossary": ["Important Entity", "Key Term", "Person's Name", "Location"],
  "confidence": 0.90
}}

CRITICAL:
- Return ONLY valid JSON
- No markdown, no extra text
- Timeline in HH:MM:SS format
- Aim for {timeline_min}-{timeline_max} timeline items

Begin analysis:"""

# INITIALIZE PHASE: Convert first segment to initial video summary
INITIALIZE_SUMMARY_PROMPT = """Convert this segment summary into an initial video summary.

VIDEO INFO:
- Video ID: {video_id}
- Title: {title}
- Duration: {duration}s

FIRST SEGMENT:
{first_segment_json}

Create initial video summary with this structure. This is just the beginning - more segments will be added.

OUTPUT (JSON only):
{{
  "summary_short": ["• Key point from this segment", "• Another point"],
  "summary_detailed": "Initial detailed summary based on first segment...",
  "timeline": [
    {{"start": "00:00:10", "end": "00:00:45", "title": "Event", "note": "Description"}}
  ],
  "glossary": ["Entity1", "Entity2"],
  "confidence": 0.85
}}

CRITICAL:
- Return ONLY valid JSON
- No markdown, no extra text
- This is segment 1 of many - keep it focused but structured

Begin initialization:"""


# STREAMING UPDATE PHASE: Add one segment to accumulated summary
STREAMING_UPDATE_PROMPT = """Update the video summary by incorporating a new segment.

VIDEO INFO:
- Video ID: {video_id}
- Title: {title}
- Progress: Segment {segment_num}/{total_segments}

CURRENT ACCUMULATED SUMMARY:
{current_summary_json}

NEW SEGMENT TO ADD:
{new_segment_json}

YOUR TASK:
Integrate the new segment into the existing summary.

INSTRUCTIONS:
- Add new information from the segment
- Merge similar points (avoid duplication)
- Keep chronological order
- Update timeline with new events
- Add new entities to glossary
- Maintain {num_bullets} bullet points (merge/replace if needed)
- Keep detailed summary under {max_words_detailed} words
- Timeline should have {timeline_min}-{timeline_max} items total
- Glossary should have {glossary_min}-{glossary_max} items total

OUTPUT FORMAT (JSON only):
{{
  "summary_short": [
    "• Updated point 1...",
    "• Updated point 2...",
    "• Updated point 3...",
    "• Updated point 4...",
    "• Updated point 5..."
  ],
  "summary_detailed": "Updated comprehensive summary integrating the new segment...",
  "timeline": [
    {{"start": "00:00:10", "end": "00:00:45", "title": "Event", "note": "Description"}},
    {{"start": "00:05:20", "end": "00:06:00", "title": "New Event", "note": "From new segment"}}
  ],
  "glossary": ["Entity1", "Entity2", "NewEntity"],
  "confidence": 0.87
}}

CRITICAL:
- Return ONLY valid JSON
- Preserve important info from both current summary and new segment
- Timestamps in HH:MM:SS format

Begin update:"""


# Helper functions


def get_map_prompt(segment_info: dict, metadata: dict, transcript: str, config) -> str:
    """Generate map phase prompt for a segment."""
    max_words = config.summarization.constraints.max_words_segment

    return MAP_PHASE_PROMPT.format(
        start_time=segment_info["start"],
        end_time=segment_info["end"],
        duration=segment_info["duration"],
        title=metadata.get("title", "Unknown"),
        topic_name=metadata.get("topic_name", "Unknown"),
        transcript_text=transcript if transcript else "No transcript available",
        max_words=max_words,
    )


def get_initialize_prompt(first_segment: dict, video_id: str, metadata: dict) -> str:
    """Generate prompt to initialize summary from first segment."""
    return INITIALIZE_SUMMARY_PROMPT.format(
        video_id=video_id,
        title=metadata.get("title", "Unknown"),
        duration=metadata.get("duration_seconds", 0),
        first_segment_json=json.dumps(first_segment, indent=2),
    )


def get_streaming_update_prompt(
    current_summary: dict,
    new_segment: dict,
    video_id: str,
    metadata: dict,
    segment_num: int,
    total_segments: int,
    config,
) -> str:
    """Generate prompt to add new segment to accumulated summary."""
    constraints = config.summarization.constraints

    return STREAMING_UPDATE_PROMPT.format(
        video_id=video_id,
        title=metadata.get("title", "Unknown"),
        segment_num=segment_num,
        total_segments=total_segments,
        current_summary_json=json.dumps(current_summary, indent=2),
        new_segment_json=json.dumps(new_segment, indent=2),
        num_bullets=constraints.summary_short_bullets,
        max_words_detailed=constraints.max_words_detailed,
        timeline_min=constraints.timeline_items_min,
        timeline_max=constraints.timeline_items_max,
        glossary_min=constraints.glossary_items_min,
        glossary_max=constraints.glossary_items_max,
    )


def get_reduce_prompt(
    video_id: str, metadata: dict, segment_summaries: list, config
) -> str:
    """Generate reduce phase prompt for merging segments."""
    constraints = config.summarization.constraints

    return REDUCE_PHASE_PROMPT.format(
        video_id=video_id,
        title=metadata.get("title", "Unknown"),
        topic_name=metadata.get("topic_name", "Unknown"),
        duration=metadata.get("duration_seconds", 0),
        num_segments=len(segment_summaries),
        segment_summaries_json=json.dumps(segment_summaries, indent=2),
        num_bullets=constraints.summary_short_bullets,
        max_words_detailed=constraints.max_words_detailed,
        timeline_min=constraints.timeline_items_min,
        timeline_max=constraints.timeline_items_max,
        glossary_min=constraints.glossary_items_min,
        glossary_max=constraints.glossary_items_max,
    )


def get_direct_prompt(video_id: str, metadata: dict, transcript: str, config) -> str:
    """Generate direct summarization prompt for short videos."""
    constraints = config.summarization.constraints

    return DIRECT_SUMMARY_PROMPT.format(
        video_id=video_id,
        title=metadata.get("title", "Unknown"),
        topic_name=metadata.get("topic_name", "Unknown"),
        duration=metadata.get("duration_seconds", 0),
        transcript_text=transcript if transcript else "No transcript available",
        num_bullets=constraints.summary_short_bullets,
        max_words_detailed=constraints.max_words_detailed,
        timeline_min=constraints.timeline_items_min,
        timeline_max=constraints.timeline_items_max,
        glossary_min=constraints.glossary_items_min,
        glossary_max=constraints.glossary_items_max,
    )
