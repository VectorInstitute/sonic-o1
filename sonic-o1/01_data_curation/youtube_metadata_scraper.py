"""YouTube Metadata Scraper for Fairness Analysis.

Collects metadata for videos across 13 topics with various demographic
dimensions.
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

load_dotenv()

# YouTube API Configuration
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


def load_config(config_path: str = None) -> Dict:
    """Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to config file. If None, searches for
            config.yaml or config.json in current directory.
    
    Returns:
        Configuration dictionary, or None if file not found.
    """
    if config_path is None:
        if os.path.exists('config.yaml'):
            config_path = 'config.yaml'
        elif os.path.exists('config.json'):
            config_path = 'config.json'
        else:
            print("ERROR: No configuration file found!")
            print(
                "Please create either config.yaml or config.json "
                "with your settings."
            )
            return None
    
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        return None
    
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except ImportError:
            print(
                "ERROR: PyYAML not installed. "
                "Install with: pip install pyyaml"
            )
            return None
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    print(f"✓ Loaded configuration from {config_path}")
    return config


CONFIG = load_config()
if CONFIG:
    API_KEY = os.environ['YT_SCRAP_API']
    BASE_DIR = CONFIG['directories']['base_dir']
    VIDEOS_DIR = os.path.join(
        BASE_DIR,
        CONFIG['directories']['videos_dir']
    )
else:
    API_KEY = None
    BASE_DIR = None
    VIDEOS_DIR = None

# Topics with their focus areas and fairness dimensions
TOPICS = {
    1: {
        "name": "Patient-Doctor Consultations",
        "search_terms": [
            "doctor patient conversation full session",
            "clinic consultation recording",
            "telehealth visit recording",
            "primary care consultation unedited",
            "medical intake interview full visit"
        ],
        "focus": "Medical communication, empathy, diagnosis discussions",
        "channel_id": None
    },
    2: {
        "name": "Job Interviews",
        "search_terms": [
            "panel interview full interview",
            "candidate interview recording",
            "on site interview hiring manager",
            "technical interview session full",
            "HR screening call recording"
        ],
        "focus": (
            "Professional interaction, emotion detection, body language"
        ),
        "channel_id": None
    },
    3: {
        "name": "Parent-Teacher Conferences",
        "search_terms": [
            "parent teacher conference recording",
            "PTC meeting full session",
            "student progress meeting recording",
            "IEP meeting full",
            "teacher parent meeting unedited"
        ],
        "focus": (
            "Educational settings, conflict resolution, child advocacy"
        ),
        "channel_id": None
    },
    4: {
        "name": "Customer Service Interactions",
        "search_terms": [
            "front desk dispute",
            "restaurant customer service",
            "Customer Service SNL",
            "employee vs customer  -karen -compilation",
            "store manager customer service",
            "angry customer store footage -staged"
        ],
        "focus": (
            "Complaint handling, emotion regulation, problem-solving"
        ),
        "channel_id": None
    },
    5: {
        "name": "Courtroom Proceedings",
        "search_terms": [
            "oral argument",
            "\"sentencing hearing\" \"full recording\" courtroom",
            "municipal court arraignment calendar session full",
            "\"sentencing hearing\" full recording courtroom",
            "small claims court full hearing official recording",
            "mock trial full"
        ],
        "focus": (
            "Legal settings, testimony analysis, fairness assessment"
        ),
        "channel_id": None
    },
    6: {
        "name": "Emergency Response Scenarios",
        "search_terms": [
            (
                "firefighters highway incident \"full response\" "
                "-shorts -news"
            ),
            "bodycam  police",
            "bodycam police rescue",
            "Law&Crime BodyCam",
            "real emergency calls paramedic"
        ],
        "focus": "Crisis management, first aid, triage decisions",
        "channel_id": None
    },
    7: {
        "name": "Public Transportation Conflicts",
        "search_terms": [
            "bus passenger fight driver -news -compilation",
            "bar fight",
            "train passenger confrontation cctv",
            "airport security passenger meltdown bodycam",
            "grocery store argument customer",
            "parking lot dispute road rage"
        ],
        "focus": (
            "Social etiquette, accessibility, conflict de-escalation"
        ),
        "channel_id": None
    },
    8: {
        "name": "Workplace Team Meetings",
        "search_terms": [
            (
                "\"team meeting\" recording Zoom -webinar "
                "-tutorial -class"
            ),
            "daily standup meeting",
            "scrum meeting real team -demo -example",
            "sprint review meeting",
            "workplace \"meeting\" recording",
        ],
        "focus": "Collaboration, leadership dynamics, idea contribution",
        "channel_id": None
    },
    9: {
        "name": "Housing/Apartment Tours",
        "search_terms": [
            "open house walkthrough agent client",
            "apartment tour with agent full",
            "rental inspection landlord tenant recording",
            "home showing buyer walkthrough",
            "accessible apartment tour elevator ramp"
        ],
        "focus": "Real estate interactions, accessibility features",
        "channel_id": None
    },
    10: {
        "name": "Restaurant Service Encounters",
        "search_terms": [
            "restaurant vlog",
            "waitress day in the life",
            "restaurant behind the scenes",
            "food service worker",
            "restaurant review visit"
        ],
        "focus": (
            "Service quality, complaint handling, accessibility"
        ),
        "channel_id": None
    },
    11: {
        "name": "Mental Health Counseling",
        "search_terms": [
            "counseling session demonstration",
            "therapy role play training",
            "mock therapy session psychology",
            "counseling techniques demonstration video",
            "therapeutic communication examples"
        ],
        "focus": (
            "Therapeutic alliance, emotional support, "
            "crisis intervention"
        ),
        "channel_id": None
    },
    12: {
        "name": "Community Town Halls",
        "search_terms": [
            "\"town hall\" \"full recording\" community Q&A",
            "town hall meeting complete",
            (
                "\"city council\" meeting \"livestream archive\" "
                "-highlights -clips"
            ),
            "community meeting local government",
            (
                "\"Islamic center\" community forum full "
                "-news -compilation"
            )
        ],
        "focus": (
            "Civic engagement, diverse viewpoints, accessibility"
        ),
        "channel_id": None
    },
    13: {
        "name": "Olympics",
        "search_terms": [
            "olympic games highlights",
            "summer olympics events",
            "winter olympics full coverage",
            "olympic moments compilation",
            "olympics replay full event"
        ],
        "focus": "Sports videos",
        "channel_id": None
    }
}

DEMOGRAPHICS = CONFIG['demographics']


def categorize_duration(duration_seconds: int) -> str:
    """Categorize video duration into short/medium/long.
    
    Args:
        duration_seconds: Duration in seconds.
    
    Returns:
        Category string: 'short', 'medium', 'long', or 'other'.
    """
    duration_minutes = duration_seconds / 60
    
    if 0.5 <= duration_minutes < 5:
        return 'short'
    elif 5 <= duration_minutes < 20:
        return 'medium'
    elif 20 <= duration_minutes <= 60:
        return 'long'
    else:
        return 'other'


class YouTubeMetadataScraper:
    """Scraper for YouTube video metadata with quality filtering."""
    
    def __init__(self, api_key: str, config: Dict = None):
        """Initialize the scraper.
        
        Args:
            api_key: YouTube Data API v3 key.
            config: Configuration dictionary.
        """
        self.youtube = build(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            developerKey=api_key
        )
        self.config = config or CONFIG
        self.rate_limit_delay = (
            self.config['api_settings'].get('rate_limit_delay', 1)
        )
        self.max_results = (
            self.config['api_settings']
            .get('max_results_per_query', 50)
        )
        self.caption_text_limit = (
            self.config['collection_settings']
            .get('caption_text_limit', 5000)
        )
        self.video_duration = (
            self.config['collection_settings']
            .get('video_duration', 'medium')
        )
        years_back = (
            self.config['search_settings'].get('years_back', 5)
        )
        self.published_after = (
            (datetime.now() - timedelta(days=years_back*365))
            .isoformat() + 'Z'
        )
        self.video_license = (
            self.config['search_settings']
            .get('video_license', 'any')
        )
        
    def search_videos(
        self,
        query: str,
        max_results: int = None,
        channel_id: str = None,
        topic_id: int = None
    ) -> List[str]:
        """Search for videos and return video IDs sorted by view count.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            channel_id: Optional channel ID to filter by.
            topic_id: Optional topic ID for special handling.
        
        Returns:
            List of video IDs.
        """
        if max_results is None:
            max_results = self.max_results
        if topic_id in [4, 6, 7, 8]:
            video_duration = 'any'
        else:
            video_duration = self.video_duration
            
        try:
            search_params = {
                'q': query,
                'part': 'id',
                'maxResults': min(max_results or self.max_results, 80),
                'type': 'video',
                'order': 'relevance',
                'relevanceLanguage': 'en',
                'videoCaption': 'any',
                'videoDefinition': 'high',
                'videoDuration': video_duration,
                'videoLicense': self.video_license,
                'publishedAfter': self.published_after
            }
            
            if channel_id:
                search_params['channelId'] = channel_id
            
            search_response = (
                self.youtube.search().list(**search_params).execute()
            )
            
            video_ids = [
                item['id']['videoId']
                for item in search_response.get('items', [])
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
        
        Returns:
            List of dictionaries containing video metadata.
        """
        if not video_ids:
            return []
        
        try:
            video_response = self.youtube.videos().list(
                part=(
                    'snippet,contentDetails,statistics,'
                    'status,topicDetails'
                ),
                id=','.join(video_ids)
            ).execute()
            
            videos_data = []
            for item in video_response.get('items', []):
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
        
        Returns:
            Structured video metadata dictionary.
        """
        snippet = item.get('snippet', {})
        content_details = item.get('contentDetails', {})
        statistics = item.get('statistics', {})
        status = item.get('status', {})
        
        video_id = item['id']
        
        duration_str = content_details.get('duration', 'PT0S')
        duration_seconds = self._parse_duration(duration_str)
        
        has_captions = content_details.get('caption') == 'true'
        caption_text = (
            self._get_caption_text(video_id) if has_captions else None
        )
        
        return {
            'video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'title': snippet.get('title', ''),
            'channel_title': snippet.get('channelTitle', ''),
            'channel_id': snippet.get('channelId', ''),
            'published_at': snippet.get('publishedAt', ''),
            'duration_seconds': duration_seconds,
            'duration_formatted': duration_str,
            'duration_category': categorize_duration(duration_seconds),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'tags': ','.join(snippet.get('tags', [])),
            'category_id': snippet.get('categoryId', ''),
            'default_language': snippet.get('defaultLanguage', ''),
            'default_audio_language': (
                snippet.get('defaultAudioLanguage', '')
            ),
            'has_captions': has_captions,
            'caption_text': caption_text,
            'is_licensed_content': (
                content_details.get('licensedContent', False)
            ),
            'copyright_notice': status.get('license', ''),
            'privacy_status': status.get('privacyStatus', ''),
            'embeddable': status.get('embeddable', False),
            'public_stats_viewable': (
                status.get('publicStatsViewable', True)
            ),
            'made_for_kids': status.get('madeForKids', False),
            'topic_categories': ','.join(
                item.get('topicDetails', {})
                .get('topicCategories', [])
            )
        }
    
    def _parse_duration(self, duration_str: str) -> int:
        """Convert ISO 8601 duration to seconds.
        
        Args:
            duration_str: Duration string (e.g., 'PT15M51S').
        
        Returns:
            Duration in seconds.
        """
        pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
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
        
        Returns:
            Caption text (truncated to limit) or None if unavailable.
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                video_id
            )
            
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                transcript = (
                    transcript_list.find_generated_transcript(['en'])
                )
            
            caption_data = transcript.fetch()
            caption_text = ' '.join(
                [entry['text'] for entry in caption_data]
            )
            return caption_text[:self.caption_text_limit]
            
        except (TranscriptsDisabled, NoTranscriptFound, Exception):
            return None
            
    def generate_search_queries(
        self, topic_info: Dict
    ) -> List[tuple]:
        """Generate demographically diverse queries.
        
        Args:
            topic_info: Dictionary containing topic search terms.
        
        Returns:
            List of (query, demographic_label) tuples.
        """
        queries = []
        base_terms = topic_info.get("search_terms", [])
        
        for term in base_terms:
            queries.append((term, "general"))
        
        def make_natural_query(
            demographic: str, term: str, dim_key: str
        ) -> str:
            """Create natural-sounding queries by demographic type."""
            
            if dim_key == "race":
                return f"{demographic} {term}"
            
            elif dim_key == "gender":
                gender_map = {
                    "Male": "man",
                    "Female": "woman"
                }
                natural_gender = gender_map.get(
                    demographic, demographic.lower()
                )
                return f"{natural_gender} {term}"
            
            elif dim_key == "age":
                age_map = {
                    "Young (18-24)": "young adult",
                    "Middle (25-39)": "middle aged",
                    "Older adults (40+)": "older adult"
                }
                natural_age = age_map.get(demographic, demographic)
                return f"{natural_age} {term}"
            
            elif dim_key == "language":
                return f"{term} {demographic}"
            
            else:
                return f"{demographic} {term}"
        
        for dim_key, demographic_values in (
            self.config['demographics'].items()
        ):
            for demographic in demographic_values:
                variations = [demographic]
                
                if demographic == "Arab":
                    variations = [
                        "Arab", "Middle Eastern", "Arabic",
                        "MENA", "Arab American"
                    ]
                elif demographic == "Indigenous":
                    variations = [
                        "Indigenous", "Native American",
                        "First Nations", "Aboriginal", "tribal"
                    ]
                
                for term in base_terms[:2]:
                    for variation in variations:
                        query = make_natural_query(
                            variation, term, dim_key
                        )
                        queries.append(
                            (query, f"{dim_key}:{demographic}")
                        )
        
        return queries
    
    def filter_quality_videos(
        self,
        videos_data: List[Dict],
        topic_id: int | None = None
    ) -> List[Dict]:
        """Research-based video quality filtering system.
        
        Args:
            videos_data: List of video metadata dictionaries.
            topic_id: Optional topic ID for special handling.
        
        Returns:
            Filtered list of quality videos.
        """
        filtered = []
        is_scarce_topic = topic_id in [4, 5, 6, 7, 8, 12]
        min_quality_threshold = 30 if is_scarce_topic else 35
                
        for video in videos_data:
            views = video.get('view_count', 0)
            likes = video.get('like_count', 0)
            comments = video.get('comment_count', 0)
            duration_seconds = video.get('duration_seconds', 0)
            title = video.get('title', '')
            description = video.get('description', '')
            channel_title = video.get('channel_title', '')
            
            if views == 0 or duration_seconds == 0:
                continue
            
            is_topic7 = (topic_id == 7)
            min_duration = 15 if is_topic7 else 30
            max_duration = 5400 if is_topic7 else 3600
            
            if not (min_duration <= duration_seconds <= max_duration):
                continue
            
            engagement_rate = (
                (likes + comments) / views if views > 0 else 0
            )
            like_ratio = likes / views if views > 0 else 0
            comment_ratio = comments / views if views > 0 else 0
            
            min_views = 100 if is_scarce_topic else 500
            if views < min_views:
                continue
            
            title_lower = title.lower()
            
            strong_clickbait_patterns = [
                r'\byou won\'?t believe\b',
                r'\bshocking truth\b',
                r'\bdoctors hate\b',
                r'\bone weird trick\b',
                r'\bwhat happens next\b',
                r'\bmind[- ]?blowing\b',
                r'\bthis is why\b',
                r'\bthe truth about\b.*\bthey don\'?t want\b',
            ]
            
            moderate_clickbait_patterns = [
                r'\bgone wrong\b',
                r'\bnumber \d+ will\b',
                r'\byou need to see\b',
                r'\bwait for it\b',
                r'\bwatch till the end\b',
            ]
            
            strong_clickbait_count = sum(
                1 for pattern in strong_clickbait_patterns
                if re.search(pattern, title_lower)
            )
            
            moderate_clickbait_count = sum(
                1 for pattern in moderate_clickbait_patterns
                if re.search(pattern, title_lower)
            )
            
            is_extreme_clickbait = (
                strong_clickbait_count >= 2 or
                (
                    strong_clickbait_count >= 1 and
                    moderate_clickbait_count >= 2
                )
            )
            
            if len(title) > 0:
                caps_ratio = (
                    sum(1 for c in title if c.isupper()) / len(title)
                )
                excessive_caps = caps_ratio > 0.55
            else:
                excessive_caps = False
            
            excessive_punctuation = (
                title.count('!') > 5 or
                title.count('?') > 5 or
                len(re.findall(r'[!?]{3,}', title)) > 0 or
                len(re.findall(r'\.{3,}', title)) > 2
            )
            
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"
                "\U0001F300-\U0001F5FF"
                "\U0001F680-\U0001F6FF"
                "\U0001F1E0-\U0001F1FF"
                "]+",
                flags=re.UNICODE
            )
            emoji_count = len(emoji_pattern.findall(title))
            excessive_emojis = emoji_count > 5
            
            spam_patterns = [
                r'\bfree money\b',
                r'\bget rich quick\b',
                r'\bclick here now\b',
                r'\bfree download\b.*\bcrack\b',
                r'\bfree robux\b',
                r'\bfree vbucks\b',
                r'\b100% working\b',
            ]
            
            has_spam = any(
                re.search(pattern, title_lower)
                for pattern in spam_patterns
            )
            
            title_words = title.split()
            very_short_title = (
                len(title_words) < 3 and
                duration_seconds > 300 and
                not is_topic7
            )
            
            has_valid_channel = (
                len(channel_title) >= 3 and
                not channel_title.replace(' ', '').isdigit() and
                channel_title.strip() != ''
            )
            
            suspicious_engagement = (
                views > 10000 and
                likes == 0 and
                comments == 0
            )
            
            very_low_engagement = (
                views > 5000 and
                engagement_rate < 0.0001
            )
            
            has_description = len(description) > 50
            
            generic_only_title = all(
                word in [
                    'video', 'clip', 'footage', 'content',
                    'new', 'best', 'top'
                ]
                for word in title_lower.split() if len(word) > 3
            )
            
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
            if excessive_caps:
                quality_score -= 10
            if excessive_punctuation:
                quality_score -= 10
            if excessive_emojis:
                quality_score -= 8
            if very_short_title:
                quality_score -= 8
            if generic_only_title:
                quality_score -= 12
            if very_low_engagement:
                quality_score -= 15
            
            hard_reject = (
                is_extreme_clickbait or
                has_spam or
                suspicious_engagement or
                not has_valid_channel or
                (excessive_caps and excessive_punctuation)
            )
            
            min_quality_threshold = 30 if is_scarce_topic else 35
            
            if not hard_reject and quality_score >= min_quality_threshold:
                video['quality_score'] = quality_score
                video['engagement_rate'] = engagement_rate
                video['like_ratio'] = like_ratio
                video['comment_ratio'] = comment_ratio
                video['clickbait_score'] = (
                    strong_clickbait_count * 2 + moderate_clickbait_count
                )
                
                filtered.append(video)
        
        return filtered
            
    def scrape_topic(
        self, topic_id: int, videos_per_query: int = None
    ) -> pd.DataFrame:
        """Scrape videos for a specific topic with demographic variations.
        
        Args:
            topic_id: Integer ID of the topic to scrape.
            videos_per_query: Optional override for videos per query.
        
        Returns:
            DataFrame containing all videos for the topic.
        """
        if videos_per_query is None:
            videos_per_query = (
                self.config['collection_settings']
                .get('videos_per_query', 5)
            )
            
        topic_info = TOPICS[topic_id]
        topic_name = topic_info["name"]
        
        print(f"\n{'='*60}")
        print(f"Scraping Topic {topic_id}: {topic_name}")
        print(f"{'='*60}")
        
        existing_video_ids = self._load_existing_video_ids(topic_id)
        print(
            f"Found {len(existing_video_ids)} existing videos "
            f"for this topic"
        )
        
        channel_id = topic_info.get("channel_id")
        if channel_id:
            print(f"Filtering to channel: {channel_id}")
        
        all_videos = []
        search_queries = self.generate_search_queries(topic_info)
        
        target_videos = (
            self.config['collection_settings']
            .get('videos_per_topic', 60)
        )
        
        videos_per_query = max(7, target_videos // len(search_queries))
        
        new_videos_count = 0
        
        for query, demographic_label in search_queries:
            print(
                f"\nSearching: {query} "
                f"(demographic: {demographic_label})"
            )
            
            video_ids = self.search_videos(
                query,
                max_results=videos_per_query,
                channel_id=channel_id,
                topic_id=topic_id
            )
            print(f"  Found {len(video_ids)} video IDs")
            
            new_video_ids = [
                vid for vid in video_ids
                if vid not in existing_video_ids
            ]
            print(
                f"  New videos (not in existing data): "
                f"{len(new_video_ids)}"
            )
            
            if new_video_ids:
                video_details = self.get_video_details(new_video_ids)
                print(
                    f"  Retrieved details for {len(video_details)} "
                    f"new videos"
                )
                
                filtered_videos = self.filter_quality_videos(
                    video_details, topic_id=topic_id
                )
                print(
                    f"  After quality filtering: "
                    f"{len(filtered_videos)} videos"
                )
                
                for video in filtered_videos:
                    video['topic_id'] = topic_id
                    video['topic_name'] = topic_name
                    video['search_query'] = query
                    video['demographic_label'] = demographic_label
                    video['focus_areas'] = topic_info["focus"]
                
                all_videos.extend(filtered_videos)
                new_videos_count += len(filtered_videos)
        
        new_df = pd.DataFrame(all_videos)
        
        if not new_df.empty:
            if self.config['search_settings'].get(
                'remove_duplicates', True
            ):
                new_df = new_df.drop_duplicates(
                    subset=['video_id'], keep='first'
                )
            
            print(f"\nNew unique videos collected: {len(new_df)}")
            
            existing_df = self._load_existing_topic_data(topic_id)
            
            if existing_df is not None and not existing_df.empty:
                combined_df = pd.concat(
                    [existing_df, new_df], ignore_index=True
                )
                print(f"Total videos after merge: {len(combined_df)}")
            else:
                combined_df = new_df
                print(
                    f"No existing data, starting fresh with "
                    f"{len(combined_df)} videos"
                )
            
            if 'quality_score' in combined_df.columns:
                combined_df = combined_df.sort_values(
                    ['quality_score', 'view_count'],
                    ascending=[False, False]
                )
            else:
                combined_df = combined_df.sort_values(
                    'view_count', ascending=False
                )
            
            print(f"Final dataset size: {len(combined_df)} videos")
            return combined_df
        else:
            print(f"\nNo new videos found for {topic_name}")
            existing_df = self._load_existing_topic_data(topic_id)
            return (
                existing_df if existing_df is not None
                else pd.DataFrame()
            )
    
    def _load_existing_video_ids(self, topic_id: int) -> set:
        """Load existing video IDs from JSON file for a topic.
        
        Args:
            topic_id: Integer ID of the topic.
        
        Returns:
            Set of video IDs that already exist in the dataset.
        """
        topic_name = TOPICS[topic_id]["name"]
        safe_name = re.sub(
            r'[^\w\s-]', '', topic_name
        ).strip().replace(' ', '_')
        
        topic_dir = os.path.join(
            VIDEOS_DIR, f"{topic_id:02d}_{safe_name}"
        )
        json_path = os.path.join(
            topic_dir, f"{safe_name}_metadata.json"
        )
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    video_ids = {
                        video['video_id'] for video in existing_data
                        if 'video_id' in video
                    }
                    return video_ids
            except Exception as e:
                print(f"Warning: Error loading existing video IDs: {e}")
                return set()
        
        return set()
    
    def _load_existing_topic_data(
        self, topic_id: int
    ) -> Optional[pd.DataFrame]:
        """Load existing topic data from JSON file.
        
        Args:
            topic_id: Integer ID of the topic.
        
        Returns:
            DataFrame if file exists, None otherwise.
        """
        topic_name = TOPICS[topic_id]["name"]
        safe_name = re.sub(
            r'[^\w\s-]', '', topic_name
        ).strip().replace(' ', '_')
        
        topic_dir = os.path.join(
            VIDEOS_DIR, f"{topic_id:02d}_{safe_name}"
        )
        json_path = os.path.join(
            topic_dir, f"{safe_name}_metadata.json"
        )
        
        if os.path.exists(json_path):
            try:
                df = pd.read_json(json_path, orient='records')
                print(
                    f"Loaded {len(df)} existing videos "
                    f"from {json_path}"
                )
                return df
            except Exception as e:
                print(f"Warning: Error loading existing data: {e}")
                return None
        
        return None
    
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
        safe_name = re.sub(
            r'[^\w\s-]', '', topic_name
        ).strip().replace(' ', '_')
        
        topic_dir = os.path.join(
            VIDEOS_DIR, f"{topic_id:02d}_{safe_name}"
        )
        os.makedirs(topic_dir, exist_ok=True)
        
        csv_path = os.path.join(
            topic_dir, f"{safe_name}_metadata.csv"
        )
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved CSV: {csv_path}")
        
        json_path = os.path.join(
            topic_dir, f"{safe_name}_metadata.json"
        )
        df.to_json(
            json_path, orient='records', indent=2, force_ascii=False
        )
        print(f"Saved JSON: {json_path}")
        
        summary = {
            'topic_id': topic_id,
            'topic_name': topic_name,
            'total_videos': len(df),
            'total_views': int(df['view_count'].sum()),
            'avg_views': int(df['view_count'].mean()),
            'median_duration_seconds': int(
                df['duration_seconds'].median()
            ),
            'videos_with_captions': int(df['has_captions'].sum()),
            'caption_percentage': (
                f"{(df['has_captions'].sum() / len(df) * 100):.1f}%"
            ),
            'demographic_distribution': (
                df['demographic_label'].value_counts().to_dict()
            ),
            'last_updated': datetime.now().isoformat()
        }
        
        if 'quality_score' in df.columns:
            summary['avg_quality_score'] = float(
                df['quality_score'].mean()
            )
            summary['quality_score_distribution'] = (
                df['quality_score'].value_counts()
                .sort_index().to_dict()
            )
        
        if 'engagement_rate' in df.columns:
            summary['avg_engagement_rate'] = float(
                df['engagement_rate'].mean()
            )
        
        summary_path = os.path.join(
            topic_dir, f"{safe_name}_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_path}")
        
        print(
            f"\nSuccessfully saved {len(df)} videos for {topic_name}"
        )


def main():
    """Main execution function."""
    if CONFIG is None:
        print("\nERROR: Could not load configuration file.")
        print(
            "Please ensure config.yaml or config.json exists "
            "in the same directory as this script."
        )
        return
    
    if  not API_KEY:
        print(
            "ERROR: Please set your YouTube Data API key "
            "in your config file"
        )
        print("\nTo get an API key:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select existing one")
        print("3. Enable YouTube Data API v3")
        print("4. Create credentials (API key)")
        print(
            "5. Update the 'api_key' field in config.yaml "
            "or config.json with your key"
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
                print(
                    f"Warning: No data collected for topic {topic_id}"
                )
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing topic {topic_id}: {e}")
            continue
    
    if all_topics_data:
        combined_df = pd.concat(all_topics_data, ignore_index=True)
        combined_path = os.path.join(
            VIDEOS_DIR, "all_topics_combined.csv"
        )
        
        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            combined_df = pd.concat(
                [existing_df, combined_df], ignore_index=True
            )
            combined_df = combined_df.drop_duplicates(
                subset=['video_id'], keep='first'
            )
        
        combined_df.to_csv(combined_path, index=False, encoding='utf-8')
        print(f"\n{'='*60}")
        print(f"Combined dataset saved: {combined_path}")
        print(f"Total videos across all topics: {len(combined_df)}")
        print(f"{'='*60}")
        
        overall_summary = {
            'total_videos': len(combined_df),
            'total_topics': 13,
            'videos_per_topic': (
                combined_df.groupby('topic_name').size().to_dict()
            ),
            'total_views': int(combined_df['view_count'].sum()),
            'videos_with_captions': int(
                combined_df['has_captions'].sum()
            ),
            'caption_percentage': (
                f"{(combined_df['has_captions'].sum() / len(combined_df) * 100):.1f}%"
            ),
            'avg_duration_seconds': int(
                combined_df['duration_seconds'].mean()
            )
        }
                
        summary_path = os.path.join(
            VIDEOS_DIR, "overall_summary.json"
        )
        with open(summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        print(f"Overall summary saved: {summary_path}")


if __name__ == "__main__":
    main()