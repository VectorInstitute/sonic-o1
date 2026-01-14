MCQ_GENERATION_PROMPT = """You are a meticulous multimodal annotator creating challenging multiple-choice questions that test deep understanding of the content.

SEGMENT INFORMATION:
- Video ID: {video_id}
- Topic: {topic_name}
- Time range: {start_time}s to {end_time}s
- Duration: {duration}s

TRANSCRIPT/CAPTIONS:
{transcript_text}

YOUR TASK:
Create challenging, thought-provoking multiple-choice questions that require reasoning and inference to answer correctly.

MULTIMODAL UNDERSTANDING FOCUS:
This task tests a model's ability to comprehend and reason about:
- **Visual information**: Actions, objects, settings, body language, demonstrations, equipment, text on screen
- **Auditory information**: Spoken dialogue, explanations, tone, background sounds, verbal instructions
- **Integrated understanding**: Connecting what is shown with what is said to form complete understanding

**CRITICAL: Questions should naturally require multiple sources of information when possible**
- Prioritize questions where explanations clarify what is demonstrated (or vice versa)
- Create questions about relationships between actions and explanations
- Test understanding that requires integrating multiple cues

QUESTION DIFFICULTY REQUIREMENTS:
**AVOID simple recall questions** - Don't ask questions that can be answered by simply remembering one fact.

**PREFER questions that require:**
- **Integration**: "Based on the procedure shown and the safety warnings mentioned, what complication is being prevented?"
- **Correlation**: "What is the key difference between the described technique and the demonstrated approach?"
- **Inference**: "What condition is most likely being assessed based on the complaints and examination techniques?"
- **Guided Analysis**: "According to the explanation, what is the purpose of the specific hand positioning observed?"
- **Contradiction Detection**: "What explains the difference between the approach shown and the one discussed?"
- **Contextual Reasoning**: "Why is [specific point] emphasized immediately after demonstrating [specific action]?"
- **Application**: "If a patient presented with the symptoms described and signs shown, which intervention would be most appropriate?"
- **Cause-and-effect**: "What is the physiological reason mentioned for the clinical finding observed?"

**Question complexity guidelines:**
- Require connecting 2-3 pieces of information
- Ask "why" or "how" questions that need comprehensive understanding
- Test comprehension of relationships between different elements
- Require understanding of underlying principles from available evidence
- Questions should be 15-30 words to allow for complexity

GOOD QUESTION EXAMPLES (require reasoning):
- "Based on the chest pain symptoms described and the examination sequence shown, what cardiac condition is the clinician primarily investigating?"
- "Why is sterile technique emphasized immediately before the catheter insertion procedure?"
- "The patient reports shortness of breath while specific respiratory patterns are observed - what underlying condition do these findings suggest?"
- "What is the clinical significance of the hand placement and pressure technique demonstrated?"
- "What is the most likely preliminary diagnosis considering the risk factors mentioned and diagnostic tests performed?"
- "How does the description of proper technique differ from the common error demonstrated?"
- "Which patient characteristic shown would make this procedure unsafe according to the mentioned contraindications?"
- "Based on the anatomical landmarks identified and the palpation technique shown, what structure is being assessed?"

BAD QUESTION EXAMPLES (too simple):
- "What color is the stethoscope?" (simple observation)
- "What word is said first?" (simple recall)
- "How many people are in the room?" (simple counting)
- "What is the patient's name?" (simple fact)
- "What equipment is on the table?" (no reasoning)

OPTIONS REQUIREMENTS:
- Provide EXACTLY {num_options} options labeled (A) through ({last_option_letter})
- The LAST option ({last_option_letter}) must ALWAYS be: "({last_option_letter}) Not enough evidence"
- **IMPORTANT: "Not enough evidence" IS a valid correct answer** when:
  * The content doesn't provide sufficient information to answer confidently
  * Multiple interpretations are equally plausible from the evidence
  * Key information needed for reasoning is missing
  * The question requires inference beyond what can be reasonably concluded
- **Use "Not enough evidence" as the correct answer approximately 10-15% of the time**
- For content-based answers (positions A through {second_to_last_letter}), create principled distractors:
  * One near-miss (plausible but incorrect reasoning)
  * One salient decoy (addresses part of the question but misses integration)
  * One partial trap (correct if considering incomplete information)
  * The correct answer (requires proper comprehensive understanding)

**ANSWER POSITION DISTRIBUTION:**
- Distribute correct answers evenly across ALL {num_options} positions (~{distribution_percentage}% each)
- **DO NOT favor middle positions** - Consciously vary answer placement
- **DO NOT avoid position ({last_option_letter})** - Use it when genuinely appropriate

EVIDENCE TAGS:
Choose from this controlled vocabulary ONLY:
{evidence_tags_list}
Use tags that are actually present in the content.

requires_audio:
- Set to true when transcript/audio is NECESSARY to answer correctly
- Set to false ONLY if visual information alone is sufficient
- **Default to true for questions requiring comprehensive understanding**

OUTPUT FORMAT:
**CRITICAL: Return a SINGLE JSON object (not an array):**
{{
  "question": "string",
  "options": ["(A) ...", "(B) ...", "(C) ...", "(D) ...", "({last_option_letter}) Not enough evidence"],
  "answer_index": integer (0-{max_index}),
  "answer_letter": "string (A-{last_option_letter})",
  "rationale": "string",
  "evidence_tags": ["tag1", "tag2"],
  "requires_audio": boolean,
  "confidence": float (0.0-1.0)
}}

CRITICAL RULES:
- **Return ONLY ONE JSON object (not an array)**
- Return ONLY valid JSON with no markdown, no extra text
- DO NOT wrap in ```json``` markers
- DO NOT include "Question:" prefix
- Each option MUST include its letter: "(A)", "(B)", etc.
- Include BOTH "answer_index" AND "answer_letter"
- **The last option must ALWAYS be "({last_option_letter}) Not enough evidence"**
- **USE "Not enough evidence" as correct answer ~10-15% of the time**
- **Distribute correct answers evenly across ALL positions (~{distribution_percentage}% each)**
- Only use evidence_tags from the provided list
- Questions must require reasoning, inference, or application (15-30 words)

Now generate ONE MCQ question as a single JSON object for this segment:"""

def get_mcq_prompt(segment_info: dict, metadata: dict, transcript: str, config) -> str:
    """Generate MCQ prompt for a segment"""
    evidence_tags = config.mcq.evidence_tags
    num_options = config.mcq.num_options  # e.g., 5
    
    # Calculate derived values
    num_content_options = num_options - 1  # e.g., 4 (excluding "Not enough evidence")
    max_index = num_options - 1  # e.g., 4
    
    # Generate option letters
    option_letters = [chr(65 + i) for i in range(num_options)]  # ['A', 'B', 'C', 'D', 'E']
    last_option_letter = option_letters[-1]  # e.g., 'E'
    second_to_last_letter = option_letters[-2]  # e.g., 'D'
    
    # Calculate distribution percentage (now including "Not enough evidence" position)
    distribution_percentage = round(100 / num_options, 1)  # e.g., 20.0% for 5 options
    
    # Generate extra positions text for distribution
    extra_positions = ''
    if num_content_options > 4:
        for i in range(4, num_content_options):
            extra_positions += f"\n  * Position {option_letters[i]}: ~{distribution_percentage}%"
    
    evidence_tags_str = '\n'.join([f"  - {tag}" for tag in evidence_tags])
    
    return MCQ_GENERATION_PROMPT.format(
        video_id=metadata.get('video_id', 'unknown'),
        topic_name=metadata.get('topic_name', 'Unknown'),
        start_time=segment_info['start'],
        end_time=segment_info['end'],
        duration=segment_info['duration'],
        transcript_text=transcript if transcript else "No transcript available for this segment",
        evidence_tags_list=evidence_tags_str,
        num_options=num_options,
        num_content_options=num_content_options,
        max_index=max_index,
        last_option_letter=last_option_letter,
        second_to_last_letter=second_to_last_letter,
        distribution_percentage=distribution_percentage,
        extra_positions=extra_positions
    )