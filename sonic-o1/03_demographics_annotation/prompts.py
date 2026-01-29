"""prompts.py.

Prompt templates for demographics annotation.

Author: SONIC-O1 Team
"""

SYSTEM_PROMPT = """You are a demographics annotation specialist for academic research.
Your task is to analyze multimodal media content (video, audio, and text captions/transcripts)
and identify the demographic characteristics of ALL individuals who appear visually or speak.

You must be objective, accurate, and avoid making assumptions based on stereotypes.

CRITICAL: Return ONLY valid JSON that can be directly parsed. No explanations outside
the JSON structure."""

MAIN_PROMPT_TEMPLATE = """Analyze this MULTIMODAL media to identify demographics of ALL people who appear visually or speak.

MEDIA INFORMATION:
- Title: {title}
- Duration: {duration_seconds} seconds
- Topic: {topic_name}

INPUTS PROVIDED:
You have access to multiple modalities for this analysis:
1. VIDEO: Visual content showing individuals (if available)
2. AUDIO: Sound/speech from individuals (may be embedded in video or separate)
3. TRANSCRIPT/CAPTIONS: Text representation of spoken content
{transcript_preview}

IMPORTANT: Use ALL available modalities together for the most accurate analysis.
Cross-reference visual, audio, and text cues to identify and confirm demographics.

---

ANALYSIS GUIDELINES BY MODALITY:

**VIDEO ANALYSIS (when available):**
- Race/Ethnicity: Primary visual assessment of facial features, skin tone
- Gender: Visual presentation (appearance, clothing, mannerisms)
- Age: Visual appearance (facial features, hair, physical characteristics)
- Language: Can support audio analysis with lip movements

**AUDIO ANALYSIS (always available in video or as separate audio):**
- Gender: Vocal pitch and timbre (deep pitch → Male, high pitch → Female)
- Age: Vocal characteristics (youthful energy vs. mature tone vs. older/quavering voice)
- Language: Primary method for identifying spoken languages and accents
- Race/Ethnicity: MAY provide supporting evidence via accent, but use LOW confidence

**TRANSCRIPT/CAPTION ANALYSIS (when available):**
- Language: Confirms which languages are spoken
- Speaker identification: Helps count unique individuals
- Context: Provides semantic understanding of the interaction
- Names/References: May help distinguish between speakers

**CROSS-MODAL VERIFICATION:**
- When you have both video and audio, verify gender and age across both modalities
- Use transcript to confirm language identification from audio
- Count unique individuals by combining visual appearances with distinct voices
- Higher confidence when multiple modalities agree

---

DEMOGRAPHIC CATEGORIES TO IDENTIFY:

1. RACE/ETHNICITY (select all that apply):
   - White: European descent appearance
   - Black: African descent appearance
   - Asian: East/Southeast/South Asian appearance
   - Indigenous: Native American/Aboriginal appearance
   - Arab: Middle Eastern/North African appearance
   - Hispanic: Latin American appearance
   - **Note:** Primarily visual assessment. Audio-only inference should have LOW confidence unless very strong accent indicators.

2. GENDER (select all that apply):
   - Male: Masculine presenting individuals OR deep vocal pitch
   - Female: Feminine presenting individuals OR high vocal pitch
   - **Use visual cues first, audio cues second**

3. AGE GROUPS (select all that apply):
   - Young (18-24): Visual appearance OR youthful voice
   - Middle (25-39): Visual appearance OR mature voice
   - Older adults (40+): Visual appearance OR older voice characteristics
   - **Combine visual and audio cues for best accuracy**

4. LANGUAGE (select all spoken):
   - Identify all languages and distinct accents heard
   - Use audio AND transcript to confirm
   - Default to ["English"] if only English is spoken

---

ANALYSIS METHODOLOGY (MULTIMODAL):

**Step 1: IDENTIFY INDIVIDUALS**
- Count unique people visible in video
- Count unique voices in audio (use transcript speaker labels if available)
- Total = unique individuals across both modalities

**Step 2: ASSESS EACH INDIVIDUAL**
For each person:
- If visible: Use video for race, gender, age (HIGH confidence)
- If speaking: Use audio for gender, age, language (MEDIUM-HIGH confidence)
- If both: Cross-verify and use HIGHEST confidence
- Use transcript to confirm language and count speakers

**Step 3: ASSIGN CONFIDENCE**
- 0.9-1.0: Clear visual evidence OR audio + visual agreement
- 0.7-0.89: Clear audio evidence OR single modality with good clarity
- 0.5-0.69: Uncertain (e.g., accent-based inference, unclear visuals)
- Below 0.5: Do not include

**Step 4: LIST UNIQUE DEMOGRAPHICS**
- Aggregate all unique demographics across all individuals
- Include only those meeting minimum confidence threshold

---

CONFIDENCE SCORING GUIDELINES:

**HIGH CONFIDENCE (0.9-1.0):**
- Clear face visible in video
- Multiple modalities agree
- Crystal clear audio with obvious characteristics

**MEDIUM CONFIDENCE (0.7-0.89):**
- Reasonable visual clarity
- Clear vocal characteristics
- Single modality but strong evidence

**LOW CONFIDENCE (0.5-0.69):**
- Distant/unclear visuals
- Accent-based inferences
- Muffled or brief audio
- Single weak indicator

---

STEP-BY-STEP REASONING (INTERNAL - DO NOT OUTPUT):

Perform this analysis internally but DO NOT include in your response:

1. **Individual Identification:**
   - List each person: "Person 1 (visible + speaking), Person 2 (voice only), Person 3 (visible only)"

2. **Per-Individual Demographics:**
   - Person 1: Race [visual], Gender [visual+audio], Age [visual+audio], Language [audio+transcript]
   - Person 2: Gender [audio], Age [audio], Language [audio+transcript]

3. **Aggregate Demographics:**
   - Unique races: [from all individuals]
   - Unique genders: [from all individuals]
   - Unique ages: [from all individuals]
   - Unique languages: [from all individuals]

4. **Confidence Assignment:**
   - For each demographic, assign confidence based on evidence quality and modality agreement

---

OUTPUT FORMAT:

Return ONLY this JSON structure with no additional text:

{{
  "demographics_detailed": {{
    "race": [list unique races observed with sufficient confidence],
    "gender": [list unique genders observed with sufficient confidence],
    "age": [list unique age groups observed with sufficient confidence],
    "language": [list languages/accents actually spoken]
  }},
  "demographics_confidence": {{
    "race": {{"race1": confidence1, "race2": confidence2}},
    "gender": {{"gender1": confidence1, "gender2": confidence2}},
    "age": {{"age1": confidence1, "age2": confidence2}},
    "language": {{"language1": confidence1}}
  }},
  "demographics_annotation": {{
    "model": "{model_name}",
    "annotated_at": "{timestamp}",
    "individuals_count": total_number_of_unique_individuals,
    "modalities_used": [list of "video", "audio", "transcript" that were available],
    "explanation": "Brief factual description combining visual, audio, and transcript observations. E.g., 'Video shows 2 individuals having a conversation. Audio reveals 2 distinct voices (1 male, 1 female). Transcript confirms English language.'"
  }}
}}

CRITICAL REMINDERS:
- Use ALL available modalities (video, audio, transcript) together
- Cross-verify demographics across modalities for higher confidence
- Return ONLY valid JSON
- No text before or after the JSON
- Empty arrays are acceptable if no confident matches found"""


def get_validation_prompt() -> str:
    """Get prompt for validating/fixing JSON output."""
    return """The following text should be valid JSON but may have formatting issues.
    Please return ONLY the corrected valid JSON with no additional text:

    {response}"""
