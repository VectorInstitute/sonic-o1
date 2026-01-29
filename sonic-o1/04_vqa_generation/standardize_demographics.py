#!/usr/bin/env python3
"""standardize_demographics.py.

Standardize demographic values across VQA JSON files to match canonical
categories in vqa_config.yaml. Maps variant race/gender/age/language
terms to canonical values and removes "Unknown" where possible.

Usage:
    python standardize_demographics.py --config vqa_config.yaml --dry-run
    python standardize_demographics.py --config vqa_config.yaml
    python standardize_demographics.py --topics 10,11 --dry-run

Author: SONIC-O1 Team
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("standardize_demographics.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration wrapper."""

    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If relative, resolved relative
            to this script's directory.

    Returns
    -------
        Config object with nested attribute access.
    """
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).parent / config_path

    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)


class DemographicsStandardizer:
    """Standardize demographics to canonical categories."""

    # Canonical categories from config
    CANONICAL_RACE = {"White", "Black", "Asian", "Indigenous", "Arab", "Hispanic"}
    CANONICAL_GENDER = {"Male", "Female"}
    CANONICAL_AGE = {"18-24", "25-39", "40+"}
    # Extended language list to include common languages in dataset
    CANONICAL_LANGUAGE = {
        "English",
        "Hindi",
        "Arabic",
        "Spanish",
        "Chinese",
        "French",
        "Korean",
        "Thai",
        "Swahili",
        "Punjabi",
        "Telugu",
        "Malay",
        "Urdu",
    }

    # Mapping rules for race/ethnicity
    RACE_MAPPINGS = {
        # Asian variants
        "south asian": "Asian",
        "east asian": "Asian",
        "southeast asian": "Asian",
        "east/southeast asian": "Asian",
        "asian american": "Asian",
        "south east asian": "Asian",
        "indian/south asian": "Asian",
        "middle eastern": "Arab",
        # Hispanic variants
        "latino": "Hispanic",
        "latina": "Hispanic",
        "latinx": "Hispanic",
        "latin american": "Hispanic",
        "spanish": "Hispanic",
        "hispanic": "Hispanic",
        # Black variants
        "african american": "Black",
        "african": "Black",
        "afro-american": "Black",
        # White variants
        "caucasian": "White",
        "european": "White",
        "european american": "White",
        # Indigenous variants
        "native american": "Indigenous",
        "aboriginal": "Indigenous",
        "first nations": "Indigenous",
        # Arab variants
        "arab american": "Arab",
        "middle-eastern": "Arab",
        # Asian - lowercase variant
        "asian": "Asian",
        # Mixed-race - map to first mentioned or most specific
        "mixed-race": "Asian",  # Review case-by-case if needed
    }

    # Mapping rules for gender
    GENDER_MAPPINGS = {
        "man": "Male",
        "woman": "Female",
        "m": "Male",
        "f": "Female",
        "female": "Female",
        "male": "Male",
    }

    # Mapping rules for age (from descriptive to numeric brackets)
    AGE_MAPPINGS = {
        "young (18-24)": "18-24",
        "young": "18-24",
        "young adult": "18-24",
        "young adults": "18-24",
        "middle (25-39)": "25-39",
        "middle": "25-39",
        "middle age": "25-39",
        "middle-aged": "25-39",
        "middle aged": "25-39",
        "middle adults": "25-39",
        "older adults (25-39)": "25-39",
        "older adults (40+)": "40+",
        "older (40+)": "40+",
        "older adults": "40+",
        "older adult": "40+",
        "older adult (40+)": "40+",
        "older adults (+40)": "40+",
        "older 40+": "40+",
        "older": "40+",
        "old": "40+",
        "senior": "40+",
        "elderly": "40+",
    }

    # Mapping rules for language
    LANGUAGE_MAPPINGS = {
        # English variants with accents - all map to English
        "English ": "English",  # Capitalized with trailing space
        "english ": "English",  # lowercase with trailing space
        "english american accent": "English",
        "english (american)": "English",
        "english (british)": "English",
        "english (british accent)": "English",
        "english (british/uk accent)": "English",
        "english (american accent)": "English",
        "english (north american accent)": "English",
        "english (north american)": "English",
        "english (uk accent)": "English",
        "english (australian accent)": "English",
        "english (indian accent)": "English",
        "english (south asian accent)": "English",
        "english (south asian/middle eastern accent)": "English",
        "english (singaporean accent)": "English",
        "english (singaporean accent), mandarin": "English",
        "english (eastern european accent)": "English",
        "english (spanish/latin american accent)": "English",
        "english (middle eastern/foreign accent)": "English",
        "english (general/non-native kenyan accent)": "English",
        "english (general/non-native accent)": "English",
        "english (non-indian accent)": "English",
        "english (jamaican accent)": "English",
        "english (caribbean/jamaican accent)": "English",
        "english (african accent)": "English",
        "english (arab accent)": "English",
        "english with accent": "English",
        "english with indian accent": "English",
        "english with indian/south asian accent": "English",
        "english with east asian accent": "English",
        "english with arab accent": "English",
        "american english": "English",
        "british english": "English",
        "english, singaporean accent": "English",
        "english, arabic accent": "English",  # with trailing space
        # Multilingual - take first/primary language
        "english, spanish": "English",
        "english, hindi": "English",
        "english, arabic": "English",
        "english, swahili": "English",
        "english, telugu": "English",
        "english, mandarin": "English",
        "english, mandarin, cantonese": "English",
        "english, hindi, punjabi": "English",
        "english, urdu": "English",
        "english, italian accent": "English",
        "french, english": "French",
        "thai, english": "Thai",
        "thai, korean, english": "English",
        "malay, english": "English",
        "malay, chinese (mandarin), english": "English",
        "mandarin, english": "Chinese",
        "hokkien, mandarin": "Chinese",
        "urdu, english": "Urdu",
        # Chinese variants
        "mandarin": "Chinese",
        "cantonese": "Chinese",
        "mandarin chinese": "Chinese",
        "chinese (mandarin)": "Chinese",
        "hokkien": "Chinese",
        # Spanish variants
        "spanish (latin american)": "Spanish",
        "spanish (spain)": "Spanish",
        "latin american spanish": "Spanish",
        # Hindi variants
        "hindi/urdu": "Hindi",
        # Arabic variants
        "modern standard arabic": "Arabic",
        "arabic (egyptian)": "Arabic",
        "arabic (levantine)": "Arabic",
        "arabic accent": "Arabic",
        # Urdu standalone
        "urdu": "Urdu",
        # Sign language - map to English (most common in dataset)
        "asl": "English",
    }

    def __init__(self, config: Config, dry_run: bool = False):
        """
        Initialize standardizer.

        Args:
            config: Configuration object
            dry_run: If True, only report what would be changed
        """
        self.config = config
        self.dry_run = dry_run

        # Statistics tracking
        self.stats = {
            "total_entries": 0,
            "entries_with_demographics": 0,
            "entries_modified": 0,
            "total_demographic_items": 0,
            "items_modified": 0,
            "race_changes": Counter(),
            "gender_changes": Counter(),
            "age_changes": Counter(),
            "language_changes": Counter(),
            "out_of_scope_race": Counter(),
            "out_of_scope_gender": Counter(),
            "out_of_scope_age": Counter(),
            "out_of_scope_language": Counter(),
            "unknown_removed": 0,
        }

    def standardize_value(self, value: str, category: str) -> Tuple[str, bool]:
        """
        Standardize a single demographic value.

        Args:
            value: Original value
            category: 'race', 'gender', 'age', or 'language'

        Returns
        -------
            Tuple of (standardized_value, was_changed)
        """
        value = value.strip()
        if not value or value.lower() == "unknown":
            return value, False

        original = value
        value_lower = value.lower()

        # Special case handling before mapping
        if category == "race":
            # Handle multi-racial entries - keep first race mentioned
            if "mixed-race" in value_lower or "mixed race" in value_lower:
                # Map to Asian as default (most common in dataset)
                pass
            elif "," in value and "white" in value_lower and "hispanic" in value_lower:
                # "White, Hispanic" -> "Hispanic" (prioritize minority)
                value_lower = "hispanic"
                mappings = self.RACE_MAPPINGS
                canonical = self.CANONICAL_RACE
                out_of_scope = self.stats["out_of_scope_race"]
                new_value = "Hispanic"
                self.stats["race_changes"][f"{original} -> {new_value}"] += 1
                return new_value, True
            elif "not specified" in value_lower:
                # Map to Unknown
                return "Unknown", True

        elif category == "age" and (
            "under" in value_lower
            or "child" in value_lower
            or value_lower.startswith("young (under")
        ):
            # Handle children - map to youngest bracket
            value_lower = "young (18-24)"

        # Select appropriate mappings
        if category == "race":
            mappings = self.RACE_MAPPINGS
            canonical = self.CANONICAL_RACE
            out_of_scope = self.stats["out_of_scope_race"]
        elif category == "gender":
            mappings = self.GENDER_MAPPINGS
            canonical = self.CANONICAL_GENDER
            out_of_scope = self.stats["out_of_scope_gender"]
        elif category == "age":
            mappings = self.AGE_MAPPINGS
            canonical = self.CANONICAL_AGE
            out_of_scope = self.stats["out_of_scope_age"]
        elif category == "language":
            mappings = self.LANGUAGE_MAPPINGS
            canonical = self.CANONICAL_LANGUAGE
            out_of_scope = self.stats["out_of_scope_language"]
        else:
            return value, False

        # Check if already canonical
        if value in canonical:
            return value, False

        # Try to map
        if value_lower in mappings:
            new_value = mappings[value_lower]
            if category == "race":
                self.stats["race_changes"][f"{original} -> {new_value}"] += 1
            elif category == "gender":
                self.stats["gender_changes"][f"{original} -> {new_value}"] += 1
            elif category == "age":
                self.stats["age_changes"][f"{original} -> {new_value}"] += 1
            elif category == "language":
                self.stats["language_changes"][f"{original} -> {new_value}"] += 1
            return new_value, True

        # Not found in mappings - track as out of scope
        if value.lower() != "unknown":
            out_of_scope[original] += 1

        return value, False

    def standardize_demographic_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Standardize a single demographic entry.

        Args:
            entry: Demographic entry dict with race, gender, age, language, count

        Returns
        -------
            True if any changes were made
        """
        changed = False

        # Standardize race
        if "race" in entry:
            new_race, race_changed = self.standardize_value(entry["race"], "race")
            if race_changed:
                entry["race"] = new_race
                changed = True

        # Standardize gender
        if "gender" in entry:
            new_gender, gender_changed = self.standardize_value(
                entry["gender"], "gender"
            )
            if gender_changed:
                entry["gender"] = new_gender
                changed = True

        # Standardize age
        if "age" in entry:
            new_age, age_changed = self.standardize_value(entry["age"], "age")
            if age_changed:
                entry["age"] = new_age
                changed = True

        # Standardize language
        if "language" in entry:
            new_language, language_changed = self.standardize_value(
                entry["language"], "language"
            )
            if language_changed:
                entry["language"] = new_language
                changed = True

        return changed

    def should_remove_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Check if demographic entry should be removed (all unknowns).

        Args:
            entry: Demographic entry dict

        Returns
        -------
            True if entry should be removed
        """
        race = entry.get("race", "").lower()
        gender = entry.get("gender", "").lower()
        age = entry.get("age", "").lower()
        language = entry.get("language", "").lower()

        # Remove if all fields are unknown
        return (
            race == "unknown"
            and gender == "unknown"
            and age == "unknown"
            and language == "unknown"
        )

    def standardize_demographics_array(
        self, demographics: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Standardize an array of demographic entries.

        Args:
            demographics: List of demographic dicts

        Returns
        -------
            Tuple of (standardized_list, was_changed)
        """
        if not demographics:
            return demographics, False

        changed = False
        standardized = []

        for entry in demographics:
            self.stats["total_demographic_items"] += 1

            # Check if should be removed
            if self.should_remove_entry(entry):
                self.stats["unknown_removed"] += 1
                changed = True
                continue

            # Standardize entry
            entry_changed = self.standardize_demographic_entry(entry)
            if entry_changed:
                self.stats["items_modified"] += 1
                changed = True

            standardized.append(entry)

        return standardized, changed

    def process_vqa_entry(self, entry: Dict[str, Any], task_name: str) -> bool:
        """
        Process a single VQA entry.

        Args:
            entry: VQA entry dict
            task_name: 'task1', 'task2', or 'task3'

        Returns
        -------
            True if entry was modified
        """
        self.stats["total_entries"] += 1

        demographics = entry.get("demographics", [])
        if not demographics:
            return False

        self.stats["entries_with_demographics"] += 1

        # Standardize demographics array
        standardized, changed = self.standardize_demographics_array(demographics)

        if changed and not self.dry_run:
            entry["demographics"] = standardized

            # Update total_individuals if field exists (task2/task3)
            if "demographics_total_individuals" in entry:
                total = sum(d.get("count", 0) for d in standardized)
                entry["demographics_total_individuals"] = total

        if changed:
            self.stats["entries_modified"] += 1

        return changed

    def process_json_file(self, json_path: Path, task_name: str) -> Dict[str, int]:
        """
        Process a single VQA JSON file.

        Args:
            json_path: Path to JSON file
            task_name: 'task1', 'task2', or 'task3'

        Returns
        -------
            Dict with stats
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {json_path.name}")
        logger.info(f"{'=' * 80}")

        # Load JSON
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
            return {"total": 0, "modified": 0}

        entries = data.get("entries", [])

        # Track changes for this file
        file_stats = {"total": len(entries), "modified": 0}

        # Process each entry
        for entry in entries:
            if self.process_vqa_entry(entry, task_name):
                file_stats["modified"] += 1

        # Save if changes were made (and not dry-run)
        if file_stats["modified"] > 0:
            if self.dry_run:
                logger.info(
                    f"[DRY-RUN] Would modify {file_stats['modified']} entries in {json_path.name}"
                )
            else:
                try:
                    # Create backup
                    backup_path = json_path.with_suffix(".json.backup_standardize")
                    with open(backup_path, "w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Created backup: {backup_path.name}")

                    # Save updated file
                    with open(json_path, "w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(
                        f"✓ Saved {file_stats['modified']} changes to {json_path.name}"
                    )
                except Exception as e:
                    logger.error(f"Failed to save {json_path}: {e}")
        else:
            logger.info(f"✓ No changes needed for {json_path.name}")

        return file_stats

    def process_all_tasks(self, topic_filter: Optional[List[int]] = None):
        """
        Process all VQA task directories.

        Args:
            topic_filter: Optional list of topic IDs to process
        """
        output_dir = Path(self.config.paths.output_dir)

        if not output_dir.exists():
            logger.error(f"Output directory not found: {output_dir}")
            return

        task_dirs = {
            "task1": output_dir / "task1_summarization",
            "task2": output_dir / "task2_mcq",
            "task3": output_dir / "task3_temporal_localization",
        }

        for task_name, task_dir in task_dirs.items():
            if not task_dir.exists():
                logger.warning(f"Task directory not found: {task_dir}")
                continue

            logger.info(f"\n{'#' * 80}")
            logger.info(f"# Processing {task_name.upper()}")
            logger.info(f"{'#' * 80}")

            # Get all JSON files
            json_files = sorted(task_dir.glob("*.json"))

            # Skip backup files
            json_files = [
                f
                for f in json_files
                if not f.name.endswith(".backup")
                and not f.name.endswith(".backup_standardize")
            ]

            # Filter by topic if specified
            if topic_filter:
                json_files = [
                    f
                    for f in json_files
                    if any(f.name.startswith(f"{tid:02d}_") for tid in topic_filter)
                ]

            logger.info(f"Found {len(json_files)} JSON files to process")

            for json_path in json_files:
                self.process_json_file(json_path, task_name)

        # Print final summary
        self.print_summary()

    def print_summary(self):
        """Print comprehensive statistics summary."""
        logger.info(f"\n{'=' * 80}")
        logger.info("STANDARDIZATION SUMMARY")
        logger.info(f"{'=' * 80}")

        logger.info("\nOVERALL STATISTICS:")
        logger.info(
            f"  Total VQA entries processed:        {self.stats['total_entries']}"
        )
        logger.info(
            f"  Entries with demographics:          {self.stats['entries_with_demographics']}"
        )
        logger.info(
            f"  Entries modified:                   {self.stats['entries_modified']}"
        )
        logger.info(
            f"  Total demographic items:            {self.stats['total_demographic_items']}"
        )
        logger.info(
            f"  Items modified:                     {self.stats['items_modified']}"
        )
        logger.info(
            f"  Unknown entries removed:            {self.stats['unknown_removed']}"
        )

        # Race changes
        if self.stats["race_changes"]:
            logger.info(
                f"\nRACE/ETHNICITY CHANGES ({sum(self.stats['race_changes'].values())} total):"
            )
            for change, count in self.stats["race_changes"].most_common():
                logger.info(f"  {change}: {count}x")

        # Gender changes
        if self.stats["gender_changes"]:
            logger.info(
                f"\nGENDER CHANGES ({sum(self.stats['gender_changes'].values())} total):"
            )
            for change, count in self.stats["gender_changes"].most_common():
                logger.info(f"  {change}: {count}x")

        # Age changes
        if self.stats["age_changes"]:
            logger.info(
                f"\nAGE CHANGES ({sum(self.stats['age_changes'].values())} total):"
            )
            for change, count in self.stats["age_changes"].most_common():
                logger.info(f"  {change}: {count}x")

        # Language changes
        if self.stats["language_changes"]:
            logger.info(
                f"\nLANGUAGE CHANGES ({sum(self.stats['language_changes'].values())} total):"
            )
            for change, count in self.stats["language_changes"].most_common():
                logger.info(f"  {change}: {count}x")

        # Out of scope values (these need manual review)
        logger.info(f"\n{'=' * 80}")
        logger.info("OUT-OF-SCOPE VALUES (Need Manual Review)")
        logger.info(f"{'=' * 80}")

        if self.stats["out_of_scope_race"]:
            logger.info("\nRACE/ETHNICITY values not in canonical list:")
            for value, count in self.stats["out_of_scope_race"].most_common():
                logger.info(f"  '{value}': {count}x")
        else:
            logger.info("\n✓ All race/ethnicity values are canonical")

        if self.stats["out_of_scope_gender"]:
            logger.info("\nGENDER values not in canonical list:")
            for value, count in self.stats["out_of_scope_gender"].most_common():
                logger.info(f"  '{value}': {count}x")
        else:
            logger.info("\n✓ All gender values are canonical")

        if self.stats["out_of_scope_age"]:
            logger.info("\nAGE values not in canonical list:")
            for value, count in self.stats["out_of_scope_age"].most_common():
                logger.info(f"  '{value}': {count}x")
        else:
            logger.info("\n✓ All age values are canonical")

        if self.stats["out_of_scope_language"]:
            logger.info("\nLANGUAGE values not in canonical list:")
            for value, count in self.stats["out_of_scope_language"].most_common():
                logger.info(f"  '{value}': {count}x")
        else:
            logger.info("\n✓ All language values are canonical")

        # Summary
        total_out_of_scope = (
            len(self.stats["out_of_scope_race"])
            + len(self.stats["out_of_scope_gender"])
            + len(self.stats["out_of_scope_age"])
            + len(self.stats["out_of_scope_language"])
        )

        logger.info(f"\n{'=' * 80}")
        if total_out_of_scope > 0:
            logger.warning(
                f"⚠ Found {total_out_of_scope} unique out-of-scope values requiring manual review"
            )
            logger.warning(
                "  Consider adding these to the mapping rules in this script"
            )
        else:
            logger.info("✓ All demographic values conform to canonical categories!")


def main():
    """Run main entry point."""
    parser = argparse.ArgumentParser(
        description="Standardize Demographics in VQA Files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vqa_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help='Comma-separated topic IDs to process (e.g., "10,11")',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without making modifications",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Parse topic filter
    topic_filter = None
    if args.topics:
        try:
            topic_filter = [int(t.strip()) for t in args.topics.split(",")]
            logger.info(f"Processing topics: {topic_filter}")
        except ValueError:
            logger.error(f"Invalid topics format: {args.topics}")
            sys.exit(1)

    # Create standardizer and run
    standardizer = DemographicsStandardizer(config, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("=" * 80)
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("=" * 80)

    standardizer.process_all_tasks(topic_filter)

    logger.info("\n✓ Done!")


if __name__ == "__main__":
    main()
