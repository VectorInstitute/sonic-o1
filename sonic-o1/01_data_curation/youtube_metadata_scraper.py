"""youtube_metadata_scraper.py.

Scrapes YouTube video metadata across multiple topics with demographic variations
and applies quality filtering based on engagement and content analysis.

Author: SONIC-O1 Team
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)


try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


load_dotenv()

# YouTube API Configuration
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"


def load_config(config_path: str = None) -> Dict:
    """Load configuration from JSON or YAML file.

    Args:
        config_path: Path to config file. If None, searches for
            config.yaml or config.json in current directory.

    Returns
    -------
        Configuration dictionary, or None if file not found.
    """
    if config_path is None:
        if os.path.exists("config.yaml"):
            config_path = "config.yaml"
        elif os.path.exists("config.json"):
            config_path = "config.json"
        else:
            print("ERROR: No configuration file found!")
            print("Please create either config.yaml or config.json with your settings.")
            return None

    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        return None

    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        if not YAML_AVAILABLE:
            print("ERROR: PyYAML not installed. Install with: pip install pyyaml")
            return None
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    print(f"✓ Loaded configuration from {config_path}")
    return config


CONFIG = load_config()
if CONFIG:
    API_KEY = os.environ["YT_SCRAP_API"]
    BASE_DIR = CONFIG["directories"]["base_dir"]
    VIDEOS_DIR = os.path.join(BASE_DIR, CONFIG["directories"]["videos_dir"])
    # Load topics from config (can be tuned/modified in config.yaml)
    TOPICS = CONFIG.get("topics", {})
else:
    API_KEY = None
    BASE_DIR = None
    VIDEOS_DIR = None
    TOPICS = {}

# Load demographics from config
DEMOGRAPHICS = CONFIG.get("demographics", {}) if CONFIG else {}


def categorize_duration(duration_seconds: int) -> str:
    """Categorize video duration into short/medium/long.

    Args:
        duration_seconds: Duration in seconds.

    Returns
    -------
        Category string: 'short', 'medium', 'long', or 'other'.
    """
    duration_minutes = duration_seconds / 60

    if 0.5 <= duration_minutes < 5:
        return "short"
    if 5 <= duration_minutes < 20:
        return "medium"
    if 20 <= duration_minutes <= 60:
        return "long"
    return "other"


class YouTubeMetadataScraper:
    """Scraper for YouTube video metadata with quality filtering."""

    def __init__(self, api_key: str, config: Dict = None):
        """Initialize the scraper.

        Args:
            api_key: YouTube Data API v3 key.
            config: Configuration dictionary.
        """
        self.youtube = build(
            YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=api_key
        )
        self.config = config or CONFIG
        self.rate_limit_delay = self.config["api_settings"].get("rate_limit_delay", 1)
        self.max_results = self.config["api_settings"].get("max_results_per_query", 50)
        self.caption_text_limit = self.config["collection_settings"].get(
            "caption_text_limit", 5000
        )
        self.video_duration = self.config["collection_settings"].get(
            "video_duration", "medium"
        )
        years_back = self.config["search_settings"].get("years_back", 5)
        self.published_after = (
            datetime.now() - timedelta(days=years_back * 365)
        ).isoformat() + "Z"
        self.video_license = self.config["search_settings"].get("video_license", "any")

    def search_videos(
        self,
        query: str,
        max_results: int = None,
        channel_id: str = None,
        topic_id: int = None,
    ) -> List[str]:
        """Search for videos and return video IDs sorted by view count.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            channel_id: Optional channel ID to filter by.
            topic_id: Optional topic ID for special handling.

        Returns
        -------
            List of video IDs.
        """
        if max_results is None:
            max_results = self.max_results
        video_duration = "any" if topic_id in [4, 6, 7, 8] else self.video_duration

        try:
            search_params = {
                "q": query,
                "part": "id",
                "maxResults": min(max_results or self.max_results, 80),
                "type": "video",
                "order": "relevance",
                "relevanceLanguage": "en",
                "videoCaption": "any",
                "videoDefinition": "high",
                "videoDuration": video_duration,
                "videoLicense": self.video_license,
                "publishedAfter": self.published_after,
            }

            if channel_id:
                search_params["channelId"] = channel_id

            search_response = self.youtube.search().list(**search_params).execute()

            video_ids = [
                item["id"]["videoId"] for item in search_response.get("items", [])
            ]
            time.sleep(self.rate_limit_delay)
            return video_ids

        except Exception as e:
            print(f"Error searching videos for query '{query}': {e}")
            return []

    def get_video_details(self, video_ids: List[str]) -> List[Dict]:
        """Get detailed metadata for a list of video IDs.

        Args:
            video_ids: List of YouTube video IDs.

        Returns
        -------
            List of dictionaries containing video metadata.
        """
        if not video_ids:
            return []

        try:
            video_response = (
                self.youtube.videos()
                .list(
                    part=("snippet,contentDetails,statistics,status,topicDetails"),
                    id=",".join(video_ids),
                )
                .execute()
            )

            videos_data = []
            for item in video_response.get("items", []):
                video_data = self._parse_video_item(item)
                videos_data.append(video_data)

            time.sleep(self.rate_limit_delay)
            return videos_data

        except Exception as e:
            print(f"Error getting video details: {e}")
            return []

    def _parse_video_item(self, item: Dict) -> Dict:
        """Parse video item from API response into structured metadata.

        Args:
            item: Video item dictionary from YouTube API.

        Returns
        -------
            Structured video metadata dictionary.
        """
        snippet = item.get("snippet", {})
        content_details = item.get("contentDetails", {})
        statistics = item.get("statistics", {})
        status = item.get("status", {})

        video_id = item["id"]

        duration_str = content_details.get("duration", "PT0S")
        duration_seconds = self._parse_duration(duration_str)

        has_captions = content_details.get("caption") == "true"
        caption_text = self._get_caption_text(video_id) if has_captions else None

        return {
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": snippet.get("title", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "channel_id": snippet.get("channelId", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration_seconds": duration_seconds,
            "duration_formatted": duration_str,
            "duration_category": categorize_duration(duration_seconds),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "tags": ",".join(snippet.get("tags", [])),
            "category_id": snippet.get("categoryId", ""),
            "default_language": snippet.get("defaultLanguage", ""),
            "default_audio_language": (snippet.get("defaultAudioLanguage", "")),
            "has_captions": has_captions,
            "caption_text": caption_text,
            "is_licensed_content": (content_details.get("licensedContent", False)),
            "copyright_notice": status.get("license", ""),
            "privacy_status": status.get("privacyStatus", ""),
            "embeddable": status.get("embeddable", False),
            "public_stats_viewable": (status.get("publicStatsViewable", True)),
            "made_for_kids": status.get("madeForKids", False),
            "topic_categories": ",".join(
                item.get("topicDetails", {}).get("topicCategories", [])
            ),
        }

    def _parse_duration(self, duration_str: str) -> int:
        """Convert ISO 8601 duration to seconds.

        Args:
            duration_str: Duration string (e.g., 'PT15M51S').

        Returns
        -------
            Duration in seconds.
        """
        pattern = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
        match = pattern.match(duration_str)

        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)

        return hours * 3600 + minutes * 60 + seconds

    def _get_caption_text(self, video_id: str) -> Optional[str]:
        """Get caption/transcript text for a video.

        Args:
            video_id: YouTube video ID.

        Returns
        -------
            Caption text (truncated to limit) or None if unavailable.
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            try:
                transcript = transcript_list.find_transcript(["en"])
            except (NoTranscriptFound, Exception):
                transcript = transcript_list.find_generated_transcript(["en"])

            caption_data = transcript.fetch()
            caption_text = " ".join([entry["text"] for entry in caption_data])
            return caption_text[: self.caption_text_limit]

        except (TranscriptsDisabled, NoTranscriptFound, Exception):
            return None

    def generate_search_queries(self, topic_info: Dict) -> List[tuple]:
        """Generate demographically diverse queries.

        Args:
            topic_info: Dictionary containing topic search terms.

        Returns
        -------
            List of (query, demographic_label) tuples.
        """
        queries = []
        base_terms = topic_info.get("search_terms", [])

        for term in base_terms:
            queries.append((term, "general"))

        def make_natural_query(demographic: str, term: str, dim_key: str) -> str:
            """Create natural-sounding queries by demographic type."""
            if dim_key == "race":
                return f"{demographic} {term}"

            if dim_key == "gender":
                gender_map = {"Male": "man", "Female": "woman"}
                natural_gender = gender_map.get(demographic, demographic.lower())
                return f"{natural_gender} {term}"

            if dim_key == "age":
                age_map = {
                    "Young (18-24)": "young adult",
                    "Middle (25-39)": "middle aged",
                    "Older adults (40+)": "older adult",
                }
                natural_age = age_map.get(demographic, demographic)
                return f"{natural_age} {term}"

            if dim_key == "language":
                return f"{term} {demographic}"

            return f"{demographic} {term}"

        for dim_key, demographic_values in self.config["demographics"].items():
            for demographic in demographic_values:
                variations = [demographic]

                if demographic == "Arab":
                    variations = [
                        "Arab",
                        "Middle Eastern",
                        "Arabic",
                        "MENA",
                        "Arab American",
                    ]
                elif demographic == "Indigenous":
                    variations = [
                        "Indigenous",
                        "Native American",
                        "First Nations",
                        "Aboriginal",
                        "tribal",
                    ]

                for term in base_terms[:2]:
                    for variation in variations:
                        query = make_natural_query(variation, term, dim_key)
                        queries.append((query, f"{dim_key}:{demographic}"))

        return queries

    def _validate_basic_video_metrics(
        self, video: Dict, is_scarce_topic: bool, is_topic7: bool
    ) -> bool:
        """Validate video has minimum required metrics.

        Args:
            video: Video metadata dictionary.
            is_scarce_topic: Whether this is a scarce topic.
            is_topic7: Whether this is topic 7.

        Returns
        -------
            True if video passes basic validation, False otherwise.
        """
        views = video.get("view_count", 0)
        duration_seconds = video.get("duration_seconds", 0)

        if views == 0 or duration_seconds == 0:
            return False

        min_duration = 15 if is_topic7 else 30
        max_duration = 5400 if is_topic7 else 3600

        if not (min_duration <= duration_seconds <= max_duration):
            return False

        min_views = 100 if is_scarce_topic else 500
        return not views < min_views

    def _detect_clickbait_patterns(self, title: str) -> tuple:
        """Detect clickbait patterns in title.

        Args:
            title: Video title.

        Returns
        -------
            Tuple of (strong_count, moderate_count, is_extreme_clickbait).
        """
        title_lower = title.lower()

        strong_clickbait_patterns = [
            r"\byou won\'?t believe\b",
            r"\bshocking truth\b",
            r"\bdoctors hate\b",
            r"\bone weird trick\b",
            r"\bwhat happens next\b",
            r"\bmind[- ]?blowing\b",
            r"\bthis is why\b",
            r"\bthe truth about\b.*\bthey don\'?t want\b",
        ]

        moderate_clickbait_patterns = [
            r"\bgone wrong\b",
            r"\bnumber \d+ will\b",
            r"\byou need to see\b",
            r"\bwait for it\b",
            r"\bwatch till the end\b",
        ]

        strong_clickbait_count = sum(
            1
            for pattern in strong_clickbait_patterns
            if re.search(pattern, title_lower)
        )

        moderate_clickbait_count = sum(
            1
            for pattern in moderate_clickbait_patterns
            if re.search(pattern, title_lower)
        )

        is_extreme_clickbait = strong_clickbait_count >= 2 or (
            strong_clickbait_count >= 1 and moderate_clickbait_count >= 2
        )

        return strong_clickbait_count, moderate_clickbait_count, is_extreme_clickbait

    def _detect_title_quality_issues(
        self, title: str, duration_seconds: int, is_topic7: bool
    ) -> Dict:
        """Detect quality issues in title.

        Args:
            title: Video title.
            duration_seconds: Video duration in seconds.
            is_topic7: Whether this is topic 7.

        Returns
        -------
            Dictionary of quality issue flags.
        """
        title_lower = title.lower()

        if len(title) > 0:
            caps_ratio = sum(1 for c in title if c.isupper()) / len(title)
            excessive_caps = caps_ratio > 0.55
        else:
            excessive_caps = False

        excessive_punctuation = (
            title.count("!") > 5
            or title.count("?") > 5
            or len(re.findall(r"[!?]{3,}", title)) > 0
            or len(re.findall(r"\.{3,}", title)) > 2
        )

        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "]+",
            flags=re.UNICODE,
        )
        emoji_count = len(emoji_pattern.findall(title))
        excessive_emojis = emoji_count > 5

        title_words = title.split()
        very_short_title = (
            len(title_words) < 3 and duration_seconds > 300 and not is_topic7
        )

        generic_only_title = all(
            word in ["video", "clip", "footage", "content", "new", "best", "top"]
            for word in title_lower.split()
            if len(word) > 3
        )

        return {
            "excessive_caps": excessive_caps,
            "excessive_punctuation": excessive_punctuation,
            "excessive_emojis": excessive_emojis,
            "very_short_title": very_short_title,
            "generic_only_title": generic_only_title,
        }

    def _detect_spam_and_suspicious_patterns(
        self,
        title: str,
        channel_title: str,
        views: int,
        likes: int,
        comments: int,
        engagement_rate: float,
    ) -> Dict:
        """Detect spam and suspicious engagement patterns.

        Args:
            title: Video title.
            channel_title: Channel title.
            views: View count.
            likes: Like count.
            comments: Comment count.
            engagement_rate: Engagement rate.

        Returns
        -------
            Dictionary of spam/suspicious pattern flags.
        """
        title_lower = title.lower()

        spam_patterns = [
            r"\bfree money\b",
            r"\bget rich quick\b",
            r"\bclick here now\b",
            r"\bfree download\b.*\bcrack\b",
            r"\bfree robux\b",
            r"\bfree vbucks\b",
            r"\b100% working\b",
        ]

        has_spam = any(re.search(pattern, title_lower) for pattern in spam_patterns)

        has_valid_channel = (
            len(channel_title) >= 3
            and not channel_title.replace(" ", "").isdigit()
            and channel_title.strip() != ""
        )

        suspicious_engagement = views > 10000 and likes == 0 and comments == 0

        very_low_engagement = views > 5000 and engagement_rate < 0.0001

        return {
            "has_spam": has_spam,
            "has_valid_channel": has_valid_channel,
            "suspicious_engagement": suspicious_engagement,
            "very_low_engagement": very_low_engagement,
        }

    def _calculate_engagement_metrics(
        self, views: int, likes: int, comments: int
    ) -> Dict:
        """Calculate engagement metrics.

        Args:
            views: View count.
            likes: Like count.
            comments: Comment count.

        Returns
        -------
            Dictionary of engagement metrics.
        """
        engagement_rate = (likes + comments) / views if views > 0 else 0
        like_ratio = likes / views if views > 0 else 0
        comment_ratio = comments / views if views > 0 else 0

        return {
            "engagement_rate": engagement_rate,
            "like_ratio": like_ratio,
            "comment_ratio": comment_ratio,
        }

    def _calculate_quality_score(
        self,
        duration_seconds: int,
        engagement_rate: float,
        views: int,
        like_ratio: float,
        has_description: bool,
        has_valid_channel: bool,
        moderate_clickbait_count: int,
        strong_clickbait_count: int,
        title_issues: Dict,
        very_low_engagement: bool,
    ) -> int:
        """Calculate numeric quality score based on multiple factors.

        Args:
            duration_seconds: Video duration in seconds.
            engagement_rate: Engagement rate.
            views: View count.
            like_ratio: Like ratio.
            has_description: Whether video has description.
            has_valid_channel: Whether channel is valid.
            moderate_clickbait_count: Count of moderate clickbait patterns.
            strong_clickbait_count: Count of strong clickbait patterns.
            title_issues: Dictionary of title quality issues.
            very_low_engagement: Whether engagement is very low.

        Returns
        -------
            Quality score (integer).
        """
        quality_score = 50

        if 120 <= duration_seconds <= 900:
            quality_score += 10
        elif 60 <= duration_seconds <= 1800:
            quality_score += 5

        if engagement_rate >= 0.02:
            quality_score += 15
        elif engagement_rate >= 0.01:
            quality_score += 10
        elif engagement_rate >= 0.005:
            quality_score += 5

        if views >= 100000:
            quality_score += 10
        elif views >= 10000:
            quality_score += 7
        elif views >= 5000:
            quality_score += 5
        elif views >= 1000:
            quality_score += 3

        if like_ratio >= 0.02:
            quality_score += 8
        elif like_ratio >= 0.01:
            quality_score += 5
        elif like_ratio >= 0.005:
            quality_score += 3

        if has_description:
            quality_score += 5
        if has_valid_channel:
            quality_score += 5

        if moderate_clickbait_count > 0:
            quality_score -= 5
        if strong_clickbait_count > 0:
            quality_score -= 10
        if title_issues["excessive_caps"]:
            quality_score -= 10
        if title_issues["excessive_punctuation"]:
            quality_score -= 10
        if title_issues["excessive_emojis"]:
            quality_score -= 8
        if title_issues["very_short_title"]:
            quality_score -= 8
        if title_issues["generic_only_title"]:
            quality_score -= 12
        if very_low_engagement:
            quality_score -= 15

        return quality_score

    def _should_hard_reject(
        self,
        is_extreme_clickbait: bool,
        spam_flags: Dict,
        title_issues: Dict,
    ) -> bool:
        """Determine if video should be immediately rejected.

        Args:
            is_extreme_clickbait: Whether title has extreme clickbait.
            spam_flags: Dictionary of spam/suspicious flags.
            title_issues: Dictionary of title quality issues.

        Returns
        -------
            True if video should be hard rejected, False otherwise.
        """
        return (
            is_extreme_clickbait
            or spam_flags["has_spam"]
            or spam_flags["suspicious_engagement"]
            or not spam_flags["has_valid_channel"]
            or (
                title_issues["excessive_caps"] and title_issues["excessive_punctuation"]
            )
        )


    def filter_quality_videos(
        self, videos_data: List[Dict], topic_id: int | None = None
    ) -> List[Dict]:
        """Research-based video quality filtering system.

        Args:
            videos_data: List of video metadata dictionaries.
            topic_id: Optional topic ID for special handling.

        Returns
        -------
            Filtered list of quality videos.
        """
        filtered = []
        is_scarce_topic = topic_id in [4, 5, 6, 7, 8, 12]
        is_topic7 = topic_id == 7
        min_quality_threshold = 30 if is_scarce_topic else 35

        for video in videos_data:
            views = video.get("view_count", 0)
            likes = video.get("like_count", 0)
            comments = video.get("comment_count", 0)
            duration_seconds = video.get("duration_seconds", 0)
            title = video.get("title", "")
            description = video.get("description", "")
            channel_title = video.get("channel_title", "")

            # Step 1: Validate basic metrics
            if not self._validate_basic_video_metrics(
                video, is_scarce_topic, is_topic7
            ):
                continue

            # Step 2: Calculate engagement metrics
            engagement_metrics = self._calculate_engagement_metrics(
                views, likes, comments
            )
            engagement_rate = engagement_metrics["engagement_rate"]
            like_ratio = engagement_metrics["like_ratio"]
            comment_ratio = engagement_metrics["comment_ratio"]

            # Step 3: Detect clickbait patterns
            strong_clickbait_count, moderate_clickbait_count, is_extreme_clickbait = (
                self._detect_clickbait_patterns(title)
            )

            # Step 4: Detect title quality issues
            title_issues = self._detect_title_quality_issues(
                title, duration_seconds, is_topic7
            )

            # Step 5: Detect spam and suspicious patterns
            spam_flags = self._detect_spam_and_suspicious_patterns(
                title, channel_title, views, likes, comments, engagement_rate
            )

            # Step 6: Check for hard rejection criteria
            if self._should_hard_reject(is_extreme_clickbait, spam_flags, title_issues):
                continue

            # Step 7: Calculate quality score
            has_description = len(description) > 50
            quality_score = self._calculate_quality_score(
                duration_seconds,
                engagement_rate,
                views,
                like_ratio,
                has_description,
                spam_flags["has_valid_channel"],
                moderate_clickbait_count,
                strong_clickbait_count,
                title_issues,
                spam_flags["very_low_engagement"],
            )

            # Step 8: Apply quality threshold and add to filtered list
            if quality_score >= min_quality_threshold:
                video["quality_score"] = quality_score
                video["engagement_rate"] = engagement_rate
                video["like_ratio"] = like_ratio
                video["comment_ratio"] = comment_ratio
                video["clickbait_score"] = (
                    strong_clickbait_count * 2 + moderate_clickbait_count
                )

                filtered.append(video)

        return filtered

    def _initialize_topic_scraping(
        self, topic_id: int, videos_per_query: int = None
    ) -> tuple:
        """Initialize topic scraping by loading info and generating queries.

        Args:
            topic_id: Integer ID of the topic to scrape.
            videos_per_query: Optional override for videos per query.

        Returns
        -------
            Tuple of (topic_info, topic_name, existing_video_ids, search_queries,
                     videos_per_query, target_videos).
        """
        if videos_per_query is None:
            videos_per_query = self.config["collection_settings"].get(
                "videos_per_query", 5
            )

        topic_info = TOPICS[topic_id]
        topic_name = topic_info["name"]

        print(f"\n{'=' * 60}")
        print(f"Scraping Topic {topic_id}: {topic_name}")
        print(f"{'=' * 60}")

        existing_video_ids = self._load_existing_video_ids(topic_id)
        print(f"Found {len(existing_video_ids)} existing videos for this topic")

        channel_id = topic_info.get("channel_id")
        if channel_id:
            print(f"Filtering to channel: {channel_id}")

        search_queries = self.generate_search_queries(topic_info)
        target_videos = self.config["collection_settings"].get("videos_per_topic", 60)
        videos_per_query = max(7, target_videos // len(search_queries))

        return (
            topic_info,
            topic_name,
            existing_video_ids,
            search_queries,
            videos_per_query,
            target_videos,
        )

    def _search_and_filter_query(
        self,
        query: str,
        demographic_label: str,
        topic_id: int,
        topic_info: Dict,
        topic_name: str,
        existing_video_ids: set,
        videos_per_query: int,
    ) -> List[Dict]:
        """Search videos for one query and filter for quality.

        Args:
            query: Search query string.
            demographic_label: Demographic label for this query.
            topic_id: Integer ID of the topic.
            topic_info: Dictionary containing topic information.
            topic_name: Name of the topic.
            existing_video_ids: Set of existing video IDs to skip.
            videos_per_query: Number of videos to retrieve per query.

        Returns
        -------
            List of filtered videos for this query.
        """
        print(f"\nSearching: {query} (demographic: {demographic_label})")

        channel_id = topic_info.get("channel_id")
        video_ids = self.search_videos(
            query,
            max_results=videos_per_query,
            channel_id=channel_id,
            topic_id=topic_id,
        )
        print(f"  Found {len(video_ids)} video IDs")

        new_video_ids = [vid for vid in video_ids if vid not in existing_video_ids]
        print(f"  New videos (not in existing data): {len(new_video_ids)}")

        if not new_video_ids:
            return []

        video_details = self.get_video_details(new_video_ids)
        print(f"  Retrieved details for {len(video_details)} new videos")

        filtered_videos = self.filter_quality_videos(video_details, topic_id=topic_id)
        print(f"  After quality filtering: {len(filtered_videos)} videos")

        for video in filtered_videos:
            video["topic_id"] = topic_id
            video["topic_name"] = topic_name
            video["search_query"] = query
            video["demographic_label"] = demographic_label
            video["focus_areas"] = topic_info["focus"]

        return filtered_videos

    def _load_existing_topic_data(self, topic_id: int) -> Optional[pd.DataFrame]:
        """Load existing topic data from JSON file.

        Args:
            topic_id: Integer ID of the topic.

        Returns
        -------
            DataFrame if file exists, None otherwise.
        """
        topic_name = TOPICS[topic_id]["name"]
        safe_name = re.sub(r"[^\w\s-]", "", topic_name).strip().replace(" ", "_")

        topic_dir = os.path.join(VIDEOS_DIR, f"{topic_id:02d}_{safe_name}")
        json_path = os.path.join(topic_dir, f"{safe_name}_metadata.json")

        if os.path.exists(json_path):
            try:
                df = pd.read_json(json_path, orient="records")
                print(f"Loaded {len(df)} existing videos from {json_path}")
                return df
            except Exception as e:
                print(f"Warning: Error loading existing data: {e}")
                return None

        return None

    def _merge_with_existing_data(
        self, new_df: pd.DataFrame, topic_id: int
    ) -> pd.DataFrame:
        """Merge new videos with existing topic data and remove duplicates.

        Args:
            new_df: DataFrame containing newly scraped videos.
            topic_id: Integer ID of the topic.

        Returns
        -------
            Merged DataFrame with duplicates removed.
        """
        if self.config["search_settings"].get("remove_duplicates", True):
            new_df = new_df.drop_duplicates(subset=["video_id"], keep="first")

        print(f"\nNew unique videos collected: {len(new_df)}")

        existing_df = self._load_existing_topic_data(topic_id)

        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Total videos after merge: {len(combined_df)}")
        else:
            combined_df = new_df
            print(f"No existing data, starting fresh with {len(combined_df)} videos")

        return combined_df

    def _sort_and_finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort by quality score and print final statistics.

        Args:
            df: DataFrame to sort and finalize.

        Returns
        -------
            Sorted DataFrame.
        """
        if "quality_score" in df.columns:
            df = df.sort_values(
                ["quality_score", "view_count"], ascending=[False, False]
            )
        else:
            df = df.sort_values("view_count", ascending=False)

        print(f"Final dataset size: {len(df)} videos")
        return df

    def scrape_topic(self, topic_id: int, videos_per_query: int = None) -> pd.DataFrame:
        """Scrape videos for a specific topic with demographic variations.

        Args:
            topic_id: Integer ID of the topic to scrape.
            videos_per_query: Optional override for videos per query.

        Returns
        -------
            DataFrame containing all videos for the topic.
        """
        (
            topic_info,
            topic_name,
            existing_video_ids,
            search_queries,
            videos_per_query,
            target_videos,
        ) = self._initialize_topic_scraping(topic_id, videos_per_query)

        all_videos = []

        for query, demographic_label in search_queries:
            filtered_videos = self._search_and_filter_query(
                query,
                demographic_label,
                topic_id,
                topic_info,
                topic_name,
                existing_video_ids,
                videos_per_query,
            )
            all_videos.extend(filtered_videos)

        new_df = pd.DataFrame(all_videos)

        if not new_df.empty:
            combined_df = self._merge_with_existing_data(new_df, topic_id)
            return self._sort_and_finalize_dataset(combined_df)

        print(f"\nNo new videos found for {topic_name}")
        existing_df = self._load_existing_topic_data(topic_id)
        return existing_df if existing_df is not None else pd.DataFrame()

    def _load_existing_video_ids(self, topic_id: int) -> set:
        """Load existing video IDs from JSON file for a topic.

        Args:
            topic_id: Integer ID of the topic.

        Returns
        -------
            Set of video IDs that already exist in the dataset.
        """
        topic_name = TOPICS[topic_id]["name"]
        safe_name = re.sub(r"[^\w\s-]", "", topic_name).strip().replace(" ", "_")

        topic_dir = os.path.join(VIDEOS_DIR, f"{topic_id:02d}_{safe_name}")
        json_path = os.path.join(topic_dir, f"{safe_name}_metadata.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    return {
                        video["video_id"]
                        for video in existing_data
                        if "video_id" in video
                    }
            except Exception as e:
                print(f"Warning: Error loading existing video IDs: {e}")
                return set()

        return set()

    def save_topic_data(self, df: pd.DataFrame, topic_id: int):
        """Save topic data as both CSV and JSON.

        Args:
            df: DataFrame containing video metadata.
            topic_id: Integer ID of the topic.
        """
        if df.empty:
            print("Warning: No data to save (empty DataFrame)")
            return

        topic_name = TOPICS[topic_id]["name"]
        safe_name = re.sub(r"[^\w\s-]", "", topic_name).strip().replace(" ", "_")

        topic_dir = os.path.join(VIDEOS_DIR, f"{topic_id:02d}_{safe_name}")
        os.makedirs(topic_dir, exist_ok=True)

        csv_path = os.path.join(topic_dir, f"{safe_name}_metadata.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved CSV: {csv_path}")

        json_path = os.path.join(topic_dir, f"{safe_name}_metadata.json")
        df.to_json(json_path, orient="records", indent=2, force_ascii=False)
        print(f"Saved JSON: {json_path}")

        summary = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "total_videos": len(df),
            "total_views": int(df["view_count"].sum()),
            "avg_views": int(df["view_count"].mean()),
            "median_duration_seconds": int(df["duration_seconds"].median()),
            "videos_with_captions": int(df["has_captions"].sum()),
            "caption_percentage": (
                f"{(df['has_captions'].sum() / len(df) * 100):.1f}%"
            ),
            "demographic_distribution": (
                df["demographic_label"].value_counts().to_dict()
            ),
            "last_updated": datetime.now().isoformat(),
        }

        if "quality_score" in df.columns:
            summary["avg_quality_score"] = float(df["quality_score"].mean())
            summary["quality_score_distribution"] = (
                df["quality_score"].value_counts().sort_index().to_dict()
            )

        if "engagement_rate" in df.columns:
            summary["avg_engagement_rate"] = float(df["engagement_rate"].mean())

        summary_path = os.path.join(topic_dir, f"{safe_name}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_path}")

        print(f"\nSuccessfully saved {len(df)} videos for {topic_name}")


def main():
    """Execute the YouTube metadata scraper."""
    if CONFIG is None:
        print("\nERROR: Could not load configuration file.")
        print(
            "Please ensure config.yaml or config.json exists "
            "in the same directory as this script."
        )
        return

    if not API_KEY:
        print("ERROR: Please set your YouTube Data API key in your config file")
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable YouTube Data API v3")
        print("4. Create credentials (API key)")
        print(
            "5. Update the 'api_key' field in config.yaml or config.json with your key"
        )
        return

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    scraper = YouTubeMetadataScraper(API_KEY, CONFIG)

    all_topics_data = []

    for topic_id in range(1, 13):
        try:
            df = scraper.scrape_topic(topic_id)

            if not df.empty:
                scraper.save_topic_data(df, topic_id)
                all_topics_data.append(df)
            else:
                print(f"Warning: No data collected for topic {topic_id}")

            time.sleep(2)

        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            continue

    if all_topics_data:
        combined_df = pd.concat(all_topics_data, ignore_index=True)
        combined_path = os.path.join(VIDEOS_DIR, "all_topics_combined.csv")

        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["video_id"], keep="first")

        combined_df.to_csv(combined_path, index=False, encoding="utf-8")
        print(f"\n{'=' * 60}")
        print(f"Combined dataset saved: {combined_path}")
        print(f"Total videos across all topics: {len(combined_df)}")
        print(f"{'=' * 60}")

        overall_summary = {
            "total_videos": len(combined_df),
            "total_topics": 13,
            "videos_per_topic": (combined_df.groupby("topic_name").size().to_dict()),
            "total_views": int(combined_df["view_count"].sum()),
            "videos_with_captions": int(combined_df["has_captions"].sum()),
            "caption_percentage": (
                f"{(combined_df['has_captions'].sum() / len(combined_df) * 100):.1f}%"
            ),
            "avg_duration_seconds": int(combined_df["duration_seconds"].mean()),
        }

        summary_path = os.path.join(VIDEOS_DIR, "overall_summary.json")
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"Overall summary saved: {summary_path}")


if __name__ == "__main__":
    main()
