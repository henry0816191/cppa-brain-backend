# YouTube C++ Videos Scraper

Scrapes YouTube C++ videos using YouTube Data API v3 and yt-dlp.

## Collection Modes

| Mode           | Collects                        | Skips if exists           |
| -------------- | ------------------------------- | ------------------------- |
| **metadata**   | JSON metadata only              | `{video_id}.json`         |
| **transcript** | JSON + transcript (VTT)         | JSON + transcript         |
| **full**       | JSON + transcript + video (MP4) | JSON + transcript + video |

Set in `yt_config.py` (`COLLECTION_MODE`) or via CLI (`--mode`).

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key: copy .env.template to .env and set YOUTUBE_API_KEY, or:
export YOUTUBE_API_KEY="your-key"

# Run with different modes
python scraper.py --mode metadata          # Fast: JSON only
python scraper.py --mode transcript        # Medium: JSON + transcripts
python scraper.py --mode full              # Slow: JSON + transcripts + videos

# Limit terms and retry failed downloads
python scraper.py --limit 10
python scraper.py --retry-downloads
```

## Prerequisites

- **YouTube Data API v3 key** ([Get here](https://console.cloud.google.com/))
- **yt-dlp** and **yt-dlp-ejs** (for transcript/full modes)
- **JavaScript runtime** for YouTube: **Deno** (recommended) or **Node.js** — required so yt-dlp can solve YouTube's n-challenge and get video formats. Without it you get "Only images are available" / "Requested format is not available".
- **FFmpeg** (for full mode if merging formats)
- **Cookies** (for transcript/full modes): if you see *"Sign in to confirm you're not a bot"*, see [Cookies (avoid bot detection)](#cookies-avoid-bot-detection) below.

## Cookies (avoid bot detection)

If yt-dlp fails with *"Sign in to confirm you're not a bot"*, pass YouTube cookies via one of these methods.

### Option 1: Use browser cookies (easiest on Mac)

1. Log into [YouTube](https://www.youtube.com) in **Safari** or **Chrome**.
2. In `.env` (or as environment variables), set:
   - **Safari:** `YT_COOKIES_BROWSER=safari`
   - **Chrome:** `YT_COOKIES_BROWSER=chrome`
3. Run the scraper. No cookie file needed; cookies are read from the browser.

### Option 2: Use a cookies file

1. Install a cookies-export extension (e.g. **Get cookies.txt** for Chrome, or **cookies.txt** for Safari).
2. Open [youtube.com](https://www.youtube.com) while logged in and export cookies in **Netscape** format (e.g. `cookies.txt`).
3. In `.env`, set the path: `YT_COOKIES_FILE=/path/to/cookies.txt`
4. Run the scraper.

Cookie files expire; re-export if the bot error returns.

## Configuration (yt_config.py)

### Key Settings

- `COLLECTION_MODE`: `"metadata"` | `"transcript"` | `"full"` (default: `"full"`)
- `CONCURRENT_FRAGMENTS`: `1` (sequential, prevents 403 errors)
- `DOWNLOAD_SLEEP_INTERVAL`: `3.0` (seconds between requests)
- `VIDEO_DOWNLOAD_DELAY`: `3.0` (seconds between videos)

### Search Terms (search_terms.py)

- `C_PLUS_PLUS_CREATORS`: Creator names to search
- `C_PLUS_PLUS_CHANNELS`: Channel IDs
- `C_PLUS_PLUS_PLAYLISTS`: Playlist IDs
- `SEARCH_TERMS`: Generic search terms

### Filters

- `MIN_DURATION_SECONDS`: `60` (1 minute minimum)
- `MAX_DURATION_SECONDS`: `14400` (4 hours maximum)
- `PUBLISHED_AFTER`: `"2022-01-01T00:00:00Z"`

## Output Structure

```
data/youtube_cpp/
├── json/{video_id}.json              # All modes
├── transcripts/{video_id}.en.vtt     # transcript/full modes
├── videos/{video_id}.mp4             # full mode only
└── progress_youtube_cpp.json         # Progress tracking
```

## Notes

- Duplicate videos automatically skipped
- Progress saved periodically (safe to interrupt)
- Videos >4 hours skipped in full mode
- Video format overridden to `best[ext=mp4][height<=720]` to reduce 403 errors
- Minimum 3s delay enforced in code (even if config is lower)
