"""demographics_expander.py.

Transform metadata demographics to VQA format with per-segment counting.
Uses human-reviewed demographics from metadata_enhanced.json.

Author: SONIC-O1 Team
"""

import json
import logging
from typing import Any, Dict, List


logger = logging.getLogger(__name__)


class DemographicsExpander:
    """Expand metadata demographics to VQA format with individual counts."""

    def __init__(self, config):
        """Initialize expander with configuration.

        Args:
            config: Configuration object
        """
        self.config = config
        self.demographics_categories = config.demographics.categories

    def build_expansion_prompt(
        self,
        metadata_demographics: Dict[str, List[str]],
        segment_info: Dict = None,
    ) -> str:
        """Build prompt for Gemini to expand demographics with counting.

        Args:
            metadata_demographics: From metadata_enhanced.json; keys race,
                gender, age, language; values lists of strings.
            segment_info: Optional dict with start/end times.

        Returns
        -------
            Prompt string for Gemini
        """
        race_list = metadata_demographics.get("race", [])
        gender_list = metadata_demographics.get("gender", [])
        age_list = metadata_demographics.get("age", [])
        language_list = metadata_demographics.get("language", [])

        ", ".join(race_list) if race_list else "Not specified"
        ", ".join(gender_list) if gender_list else "Not specified"
        ", ".join(age_list) if age_list else "Not specified"
        ", ".join(language_list) if language_list else "Not specified"

        segment_context = ""
        if segment_info:
            seg_start = segment_info["start"]
            seg_end = segment_info["end"]
            segment_context = (
                f"\nThis is a SEGMENT from {seg_start}s to {seg_end}s.\n"
                "Only count individuals visible/audible in THIS segment.\n"
            )

        return f"""You are analyzing video content for demographics annotation.

                HUMAN-REVIEWED DEMOGRAPHICS (Ground Truth):
                The video contains individuals with these demographic characteristics:
                - Race/Ethnicity: {", ".join(race_list) if race_list else "Not specified"}
                - Gender: {", ".join(gender_list) if gender_list else "Not specified"}
                - Age: {", ".join(age_list) if age_list else "Not specified"}
                - Language: {", ".join(language_list) if language_list else "Not specified"}

                {segment_context}

                YOUR TASK:
                1. Analyze the provided video/audio content carefully
                2. Count how many UNIQUE INDIVIDUALS with each demographic combination appear
                3. You MUST use ONLY the demographic values provided above from human review
                4. If multiple demographic combinations exist, create separate entries for each

                STRICT RULES - READ CAREFULLY:
                - Use ONLY the exact demographic values listed above (race, gender, age, language)
                - DO NOT use "Unknown" unless the human review specifies it
                - If you cannot see individuals clearly but can hear them, use the audio cues and provided demographics
                - The demographics above are human-reviewed ground truth - trust them over what you perceive
                - If the segment shows the same people from the metadata, use those exact demographic values
                - DO NOT invent new demographic categories or values

                COUNTING RULES:
                - Count UNIQUE individuals only (same person appearing multiple times = count once)
                - If you see/hear 2 males speaking, both with demographics from above, count as one entry with count: 2
                - If demographics vary across individuals, create separate entries
                - Example: 1 person with demographics A + 1 person with demographics B = two separate entries

                WHEN TO USE "Unknown":
                - ONLY use "Unknown" for a specific attribute if:
                * The human review explicitly lists "Unknown" for that attribute
                * OR you are absolutely certain there are additional individuals beyond those in the human review
                - If in doubt, use the human-reviewed demographics provided above

                OUTPUT FORMAT (JSON):
                {{
                "demographics": [
                    {{
                    "race": "Arab",
                    "gender": "Male",
                    "age": "Young (18-24)",
                    "language": "English",
                    "count": 2
                    }},
                    {{
                    "race": "White",
                    "gender": "Female",
                    "age": "Middle (25-39)",
                    "language": "English",
                    "count": 1
                    }}
                ],
                "total_individuals": 3,
                "confidence": 0.85,
                "explanation": "Brief explanation of what you observed and how you counted (2-3 sentences max)"
                }}

                CRITICAL REQUIREMENTS:
                - Use EXACT category values from the human-reviewed demographics above
                - Keep age categories AS-IS: "Young (18-24)", "Middle (25-39)", "Older adults (40+)"
                - DO NOT create entries with "Unknown" unless absolutely necessary
                - If segment shows people from the video, they almost certainly match the human-reviewed demographics
                - The sum of all "count" values should equal "total_individuals"
                - Return ONLY valid JSON, no additional text or markdown
                - Confidence should be 0.8+ if you're using the provided demographics correctly

                Begin analysis:"""

    def parse_demographics_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's response and validate demographics format.

        Args:
            response_text: JSON response from Gemini

        Returns
        -------
            Parsed and validated demographics dict
        """
        try:
            # Clean response
            response_text = response_text.strip()

            # Remove markdown code blocks if present
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.rfind("```")
                if end > start:
                    response_text = response_text[start:end]
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.rfind("```")
                if end > start:
                    response_text = response_text[start:end]

            # Parse JSON
            data = json.loads(response_text.strip())

            # Validate structure
            if "demographics" not in data:
                logger.error("Missing 'demographics' field in response")
                return self._get_empty_demographics()

            if not isinstance(data["demographics"], list):
                logger.error("'demographics' field is not a list")
                return self._get_empty_demographics()

            # Validate each demographic entry
            validated_demographics = []
            total_count = 0
            unknown_count = 0

            for entry in data["demographics"]:
                if not isinstance(entry, dict):
                    continue

                # Ensure required fields
                if "count" not in entry:
                    logger.warning("Missing 'count' in entry: %s", entry)
                    continue

                # Convert count to int if needed
                try:
                    count = int(entry["count"])
                except (ValueError, TypeError):
                    logger.warning("Invalid count: %s", entry.get("count"))
                    count = 0

                # Check for "Unknown" usage
                race = entry.get("race", "Unknown")
                gender = entry.get("gender", "Unknown")
                age = entry.get("age", "Unknown")
                language = entry.get("language", "Unknown")

                # Count how many unknowns in this entry
                if (
                    race == "Unknown"
                    or gender == "Unknown"
                    or age == "Unknown"
                    or language == "Unknown"
                ):
                    unknown_count += 1
                    logger.warning("Entry contains Unknown: %s", entry)

                validated_entry = {
                    "race": race,
                    "gender": gender,
                    "age": age,
                    "language": language,
                    "count": count,
                }

                validated_demographics.append(validated_entry)
                total_count += count

            # Log warning if too many unknowns
            if unknown_count > 0:
                logger.warning(
                    "%d entries contain Unknown - model may be conservative",
                    unknown_count,
                )

            confidence = float(data.get("confidence", 0.0))

            # Reduce confidence if too many unknowns
            if unknown_count > len(validated_demographics) / 2:
                logger.warning(">50%% entries have Unknown, reducing confidence")
                confidence *= 0.5

            return {
                "demographics": validated_demographics,
                "total_individuals": data.get("total_individuals", total_count),
                "confidence": confidence,
                "explanation": data.get("explanation", ""),
            }

        except json.JSONDecodeError as e:
            logger.error("Failed to parse demographics JSON: %s", e)
            logger.debug("Response: %s...", response_text[:500])
            return self._get_empty_demographics()
        except Exception as e:
            logger.error("Error parsing demographics: %s", e)
            return self._get_empty_demographics()

    def _get_empty_demographics(self) -> Dict[str, Any]:
        """Return empty demographics structure."""
        return {
            "demographics": [],
            "total_individuals": 0,
            "confidence": 0.0,
            "explanation": "Failed to parse demographics",
        }

    def merge_segment_demographics(
        self, segment_demographics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge demographics from multiple segments for video-level summary.

        Uses maximum count seen across all segments (conservative approach).

        Args:
            segment_demographics: List of demographics dicts from each segment

        Returns
        -------
            Merged demographics dict
        """
        # Track unique demographic combinations and their max counts
        demographic_map = {}
        total_max = 0
        all_explanations = []
        min_confidence = 1.0

        for seg_demo in segment_demographics:
            for entry in seg_demo.get("demographics", []):
                # Create key from demographic attributes
                key = (
                    entry.get("race", "Unknown"),
                    entry.get("gender", "Unknown"),
                    entry.get("age", "Unknown"),
                    entry.get("language", "Unknown"),
                )

                count = entry.get("count", 0)

                # Keep maximum count for this combination
                if key not in demographic_map or count > demographic_map[key].get(
                    "count", 0
                ):
                    demographic_map[key] = entry.copy()

            # Track total
            total_max = max(total_max, seg_demo.get("total_individuals", 0))

            # Collect explanations
            if seg_demo.get("explanation"):
                all_explanations.append(seg_demo["explanation"])

            # Track minimum confidence
            min_confidence = min(min_confidence, seg_demo.get("confidence", 1.0))

        # Convert back to list
        merged_demographics = list(demographic_map.values())

        return {
            "demographics": merged_demographics,
            "total_individuals": total_max,
            "confidence": min_confidence,
            "explanation": (
                "Merged from %d segments. " % len(segment_demographics)
                + " | ".join(all_explanations[:2])
            ),
        }
