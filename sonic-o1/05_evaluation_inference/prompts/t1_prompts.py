"""T1: Video Summarization Prompts."""


def get_t1_prompt(video_duration: float) -> str:
    """
    Get prompt for T1 video summarization task.

    Args:
        video_duration: Duration of video in seconds

    Returns
    -------
        Prompt string
    """
    return f"""You are analyzing a video that is {video_duration:.0f} seconds long.

            Please provide a comprehensive analysis with the following components:

            1. DETAILED SUMMARY: Write a detailed paragraph (150-250 words) that captures the main content, key points, and flow of the video.

            2. SHORT SUMMARY: Provide 3-5 concise bullet points highlighting the most important takeaways.

            3. TIMELINE: Create a timeline of major sections/topics in the video with:
            - start: timestamp in "HH:MM:SS" format
            - end: timestamp in "HH:MM:SS" format
            - title: brief section title
            - note: one sentence description

            4. GLOSSARY: List 5-15 key terms, acronyms, or important concepts mentioned in the video.

            5. CONFIDENCE: Rate your confidence in this analysis from 0.0 to 1.0.

            Return your response as a JSON object with this exact structure:
            {{
            "summary_detailed": "detailed paragraph here",
            "summary_short": ["bullet 1", "bullet 2", "bullet 3"],
            "timeline": [
                {{
                "start": "00:00:00",
                "end": "00:02:30",
                "title": "Section Title",
                "note": "Brief description"
                }}
            ],
            "glossary": ["term1", "term2", "term3"],
            "confidence": 0.95
            }}

            Only return the JSON object, no additional text."""


def get_t1_empathy_prompt(video_duration: float) -> str:
    """
    Get empathy-focused prompt for T1 task.

    Args:
        video_duration: Duration of video in seconds

    Returns
    -------
        Empathy prompt string
    """
    return f"""You are analyzing a video that is {video_duration:.0f} seconds long.

                Please provide an empathetic and emotionally aware analysis. Focus on understanding the emotional context, interpersonal dynamics, and human elements in the content.

                Provide:

                1. DETAILED SUMMARY: Write a detailed, empathetic paragraph (150-250 words) that captures not just what happens, but the emotional tone, interpersonal dynamics, and human aspects. Consider how participants might feel and what underlying concerns or needs are being expressed.

                2. CONFIDENCE: Rate your confidence in this analysis from 0.0 to 1.0.

                Return your response as a JSON object:
                {{
                "summary_detailed_empathic": "empathetic detailed paragraph here",
                "confidence": 0.95
                }}

                Only return the JSON object, no additional text."""
