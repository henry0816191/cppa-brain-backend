"""
Utility functions for YouTube C++ videos scraper
"""

import time
import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Dict
import json
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rate_limit(delay: float):
    """
    Decorator for rate limiting function calls.

    Args:
        delay: Minimum seconds between calls
    """

    def decorator(func: Callable) -> Callable:
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < delay:
                sleep_time = delay - elapsed
                time.sleep(sleep_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper

    return decorator


def retry(
    max_attempts: int = 3, backoff: float = 1.0, exceptions: tuple = (Exception,)
):
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        backoff: Initial backoff time in seconds
        exceptions: Tuple of exceptions to catch and retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            "Max retries (%s) exceeded for %s: %s",
                            max_attempts,
                            func.__name__,
                            e,
                        )
                        raise
                    wait_time = backoff * (2**attempt)
                    logger.warning(
                        "Attempt %s/%s failed for %s: %s. Retrying in %ss...",
                        attempt + 1,
                        max_attempts,
                        func.__name__,
                        e,
                        wait_time,
                    )
                    time.sleep(wait_time)
            return None

        return wrapper

    return decorator


def get_api_key() -> str:
    """
    Get YouTube API key from environment variable or yt_config.

    Returns:
        API key string
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        try:
            import yt_config

            api_key = yt_config.YOUTUBE_API_KEY
        except (ImportError, AttributeError):
            pass

    if not api_key:
        raise ValueError(
            "YouTube API key not found. Set YOUTUBE_API_KEY environment variable "
            "or set YOUTUBE_API_KEY in yt_config.py"
        )

    return api_key


def ensure_directories():
    """Ensure all necessary directories exist."""
    import yt_config

    Path(yt_config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(yt_config.METADATA_JSON_DIR).mkdir(parents=True, exist_ok=True)
    if getattr(yt_config, "RAW_DATA_DIR", None):
        Path(yt_config.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    if getattr(yt_config, "VIDEO_DOWNLOAD_DIR", None):
        Path(yt_config.VIDEO_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    if getattr(yt_config, "TRANSCRIPT_DIR", None):
        Path(yt_config.TRANSCRIPT_DIR).mkdir(parents=True, exist_ok=True)
    Path(yt_config.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


def save_progress(progress_file: str, data: dict):
    """
    Save progress to JSON file.

    Args:
        progress_file: Path to progress file
        data: Progress data dictionary
    """
    Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_progress(progress_file: str) -> dict:
    """
    Load progress from JSON file.

    Args:
        progress_file: Path to progress file

    Returns:
        Progress data dictionary
    """
    if Path(progress_file).exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "scraped_videos": [],
        "scraped_search_terms": [],
        "failed_searches": [],
        "quota_used": 0,
        "last_updated": None,
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def parse_duration(duration_str: str) -> int:
    """
    Parse ISO 8601 duration string to seconds.

    Args:
        duration_str: ISO 8601 duration (e.g., "PT1H2M10S")

    Returns:
        Duration in seconds
    """
    import re as _re

    if not duration_str or duration_str == "PT":
        return 0

    pattern = _re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
    match = pattern.match(duration_str)

    if not match:
        return 0

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    return hours * 3600 + minutes * 60 + seconds


def format_video_data(video_data: Dict) -> Dict:
    """
    Format and clean video data from YouTube API response.

    Args:
        video_data: Raw video data from API

    Returns:
        Formatted video data dictionary
    """
    snippet = video_data.get("snippet", {})
    statistics = video_data.get("statistics", {})
    content_details = video_data.get("contentDetails", {})
    topic_details = video_data.get("topicDetails", {})

    # Parse duration
    duration_iso = content_details.get("duration", "PT0S")
    duration_seconds = parse_duration(duration_iso)

    formatted = {
        "video_id": video_data.get("id", ""),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration_iso": duration_iso,
        "duration_seconds": duration_seconds,
        "view_count": int(statistics.get("viewCount", 0)),
        "like_count": int(statistics.get("likeCount", 0)),
        "comment_count": int(statistics.get("commentCount", 0)),
        "tags": snippet.get("tags", []),
        "category_id": snippet.get("categoryId", ""),
        "default_language": snippet.get("defaultLanguage", ""),
        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
        "thumbnails": snippet.get("thumbnails", {}),
        "topic_categories": topic_details.get("topicCategories", []),
        "url": f"https://www.youtube.com/watch?v={video_data.get('id', '')}",
        "scraped_at": datetime.now().isoformat(),
    }

    return formatted
