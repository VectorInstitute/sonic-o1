"""T2: Question Answering Prompts."""


def get_t2_prompt(question: str, options: list) -> str:
    """
    Get prompt for T2 multiple choice question answering task.

    Args:
        question: The question text
        options: List of answer options (e.g., ["(A) ...", "(B) ...", ...])

    Returns
    -------
        Prompt string
    """
    options_text = "\n".join(options)

    return f"""You are analyzing a video segment to answer a multiple choice question.

            QUESTION:
            {question}

            OPTIONS:
            {options_text}

            Please analyze the video content carefully and:
            1. Select the best answer from the options
            2. Provide a clear rationale explaining why this answer is correct based on what you observed in the video
            3. Rate your confidence in this answer from 0.0 to 1.0

            Return your response as a JSON object with this exact structure:
            {{
            "answer_letter": "B",
            "answer_index": 1,
            "rationale": "Explanation of why this answer is correct based on video content",
            "confidence": 0.90
            }}

            Notes:
            - answer_letter should be "A", "B", "C", "D", or "E"
            - answer_index should be 0, 1, 2, 3, or 4 (corresponding to A, B, C, D, E)
            - rationale should reference specific details from the video
            - CRITICAL: include all fields in the JSON and ensure proper formatting espicially answer_letter

            Only return the JSON object, no additional text."""
