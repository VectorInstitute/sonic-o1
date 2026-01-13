def get_t3_prompt(questions: list, segment_start: float, segment_end: float) -> str:
    """
    Build a clear, general prompt for temporal localization (T3) tasks.
    Handles any number of questions, with clear structure and flexible output.
    """
    segment_duration = segment_end - segment_start
    num_questions = len(questions)
    
    questions_text = "\n".join([
        f"{q['question_id']}. {q['question']}"
        for q in questions
    ])
    
    # Show the exact IDs expected in output
    question_ids_list = ", ".join([f'"{q["question_id"]}"' for q in questions])
    
    prompt = f"""You are analyzing a video segment from {segment_start:.1f}s to {segment_end:.1f}s (duration: {segment_duration:.1f}s).

                QUESTIONS ({num_questions} total):
                {questions_text}

                REQUIRED OUTPUT FORMAT:
                {{
                "questions": [
                    {{
                    "question_id": "001",
                    "start_s": <timestamp_float>,
                    "end_s": <timestamp_float>,
                    "confidence": <float_between_0_and_1>,
                    "rationale_model": "<your explanation>"
                    }},
                    {{
                    "question_id": "002",
                    "start_s": <timestamp_float>,
                    "end_s": <timestamp_float>,
                    "confidence": <float_between_0_and_1>,
                    "rationale_model": "<your explanation>"
                    }}
                    ... (continue for all {num_questions} questions)
                ]
                }}

                RATIONALE REQUIREMENTS:
                Explain: When E1 (anchor) occurs → When E2 (target) starts/ends  -> Temporal relationship  -> Visual/audio cues

                Example: "E1 (anchor) starts at 5.2s with speaker's introduction. E2 (target) starts at 35.0s when he says 'I am a final year medical student', ends at 36.6s. Relationship: 'after'."

                CONSTRAINTS:
                - All timestamps within [{segment_start:.1f}s, {segment_end:.1f}s]
                - start_s < end_s for each question
                - Include ALL {num_questions} questions with their question_id field
                - CRITICAL: include all fields in the JSON and ensure proper formatting
                - CRITICAL: Your response MUST include these exact question_ids: {question_ids_list} and include the question_id field
                - Return ONLY valid JSON (no markdown, no code blocks, no extra text)
                """
    return prompt