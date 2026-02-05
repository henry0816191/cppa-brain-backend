"""
YouTube preprocessing pipeline for LangChain RAG

Processes YouTube video transcripts (VTT format) and metadata (JSON format)
to create documents for RAG indexing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from tqdm import tqdm

from config import PineconeConfig, YouTubeConfig
from preprocessor.utility import (
    extract_video_id_from_filename,
    get_timestamp_from_date,
    normalize_metadata_value,
    safe_int,
    time_to_seconds,
    truncate_text,
    validate_content_length,
)

logger = logging.getLogger(__name__)


class YouTubePreprocessor:
    """Process YouTube video transcripts and metadata for RAG"""

    def __init__(self, youtube_config: Optional[YouTubeConfig] = None):
        """
        Initialize YouTube preprocessor.

        Args:
            youtube_config: YouTubeConfig instance (loads from env vars if not provided)
        """
        self.config = youtube_config or YouTubeConfig()
        self.transcripts_dir = Path(self.config.transcripts_dir)
        self.metainfo_json_dir = Path(self.config.metainfo_json_dir)
        self.scripts_ids = set()
        self.metainfo_ids = set()
        self.failed_video_ids = set()

        # Get chunking configuration
        pinecone_config = PineconeConfig()
        self.chunk_size = int(pinecone_config.chunk_size * 0.9)
        self.chunk_overlap = pinecone_config.chunk_overlap

    def load_documents(
        self,
        limit: Optional[int] = None,
        video_ids: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load and process YouTube videos as documents.

        Args:
            limit: Optional limit on number of videos to process
            video_ids: Optional list of specific video IDs to process

        Returns:
            List of Document objects with transcript content and metadata
        """
        logger.info("Loading YouTube documents...")

        # Get all available video IDs
        self._load_video_ids()

        available_video_ids = list(self.scripts_ids)
        if video_ids:
            # Filter to only requested video IDs
            available_video_ids = [vid for vid in self.scripts_ids if vid in video_ids]
            logger.info(f"Filtered to {len(available_video_ids)} requested video IDs")

        if limit and len(available_video_ids) > limit:
            available_video_ids = available_video_ids[:limit]
            logger.info(f"Limited to {limit} videos")

        documents = []
        failed_count = 0

        for video_id in tqdm(available_video_ids, desc="Processing YouTube videos"):
            try:
                video_docs = self._process_video(video_id)
                if video_docs:
                    documents.extend(video_docs)
                else:
                    failed_count += 1
                    self.failed_video_ids.add(video_id)
            except Exception as e:
                logger.warning(f"Failed to process video {video_id}: {e}")
                failed_count += 1

        logger.info(
            f"Loaded {len(documents)} YouTube documents "
            f"({failed_count} failed or skipped)"
        )
        return documents

    def _load_video_ids(self):
        """Get list of available video IDs from transcript files."""
        # Get video IDs from transcript files
        if self.transcripts_dir.exists():
            for vtt_file in self.transcripts_dir.glob("*.vtt"):
                # Extract video ID from filename
                # Format: {VIDEO_ID}.en.vtt or {VIDEO_ID}.vtt
                video_id = extract_video_id_from_filename(vtt_file.name)
                if video_id:
                    self.scripts_ids.add(video_id)

        # Also check JSON metadata files
        if self.metainfo_json_dir.exists():
            for json_file in self.metainfo_json_dir.glob("*.json"):
                video_id = json_file.stem  # Filename without extension
                self.metainfo_ids.add(video_id)

    def _process_video(self, video_id: str) -> List[Document]:
        """
        Process a single video: load transcript and metadata and return Document objects.

        Returns:
            List of Document objects, one for each merged segment.
        """
        metadata = self._load_metadata(video_id)
        if not metadata:
            logger.debug(f"No metadata found for video {video_id}")
            return []

        segments = self._load_transcript(video_id)
        if not segments:
            logger.debug(f"No transcript found for video {video_id}")
            return []

        merged_documents = self._merge_segments_to_documents(segments)
        if not merged_documents:
            return []

        timestamp = get_timestamp_from_date(metadata.get("published_at", ""))
        documents = []
        for i, doc_data in enumerate(merged_documents):
            content = self._build_segment_content(metadata, doc_data)
            if not validate_content_length(content, min_length=50):
                continue
            doc_metadata = self._build_segment_metadata(
                video_id, metadata, doc_data, i, timestamp
            )
            documents.append(Document(page_content=content, metadata=doc_metadata))
        return documents

    def _build_segment_content(
        self, metadata: Dict[str, Any], doc_data: Dict[str, Any]
    ) -> str:
        """Build page content string for one segment from metadata and transcript."""
        content_parts = []
        if metadata.get("title"):
            content_parts.append(f"Title: {metadata['title']}")
        if metadata.get("channel_title"):
            content_parts.append(f"Channel Title: {metadata['channel_title']}")
        if metadata.get("search_term"):
            content_parts.append(f"Search Term: {metadata['search_term']}")
        if metadata.get("description"):
            desc = truncate_text(metadata["description"], max_length=500)
            content_parts.append(f"Description: {desc}")
        content_parts.append(f"\nTranscript:\n{doc_data['text']}")
        return "\n\n".join(content_parts)

    def _build_segment_metadata(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        doc_data: Dict[str, Any],
        segment_index: int,
        published_at: float,
    ) -> Dict[str, Any]:
        """Build metadata dict for one segment document."""
        start_time = doc_data["start_time"]
        return {
            "doc_id": f"{video_id}_{segment_index}",
            "type": "youtube_video",
            "video_id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}&t={int(start_time)}",
            "channel_title": normalize_metadata_value(
                metadata.get("channel_title", "")
            ),
            "published_at": published_at,
            "duration_seconds": safe_int(metadata.get("duration_seconds", 0)),
            "view_count": safe_int(metadata.get("view_count", 0)),
            "like_count": safe_int(metadata.get("like_count", 0)),
            "search_term": normalize_metadata_value(metadata.get("search_term", "")),
            "start_time": start_time,
            "end_time": doc_data["end_time"],
        }

    def _load_metadata(self, video_id: str) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if video_id not in self.metainfo_ids:
            return None

        json_file = self.metainfo_json_dir / f"{video_id}.json"
        if not json_file.exists():
            return None
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Load transcript from VTT file and parse into time-stamped segments.

        Args:
            video_id: YouTube video ID

        Returns:
            List of dictionaries with 'start_time', 'end_time', and 'text' keys
        """
        # Try different possible filenames
        possible_files = [
            self.transcripts_dir / f"{video_id}.en.vtt",
            self.transcripts_dir / f"{video_id}.vtt",
        ]

        for vtt_file in possible_files:
            if vtt_file.exists():
                return self._parse_vtt_file(vtt_file)

        return []

    def _parse_vtt_file(self, vtt_file: Path) -> List[Dict[str, Any]]:
        """
        Parse VTT file and extract time-stamped segments.

        Pattern: line_1 = timestamp (with align:start position:0%), line_2 = script,
        line_3 = empty. Returns list of dicts with 'start_time', 'end_time', 'text'.
        """
        try:
            with open(vtt_file, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError as e:
            logger.error("Error reading VTT file %s: %s", vtt_file, e)
            return []

        segments = []
        i = 0
        while i < len(lines) - 2:
            line_1 = lines[i].strip()
            line_2 = lines[i + 1].strip()
            line_3 = lines[i + 2].strip()
            if "-->" not in line_1:
                i += 1
                continue
            if not line_2:
                i += 2
                continue
            if line_3 and "align:start position:0%" in line_1:
                i += 3
                continue
            segment = self._parse_one_vtt_cue(line_1, line_2, line_3)
            if segment:
                segments.append(segment)
            i += 3
        return segments

    def _parse_one_vtt_cue(
        self, line_1: str, line_2: str, line_3: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse one VTT cue (timestamp line, content line, optional third line).
        Returns segment dict or None if timestamp does not match.
        """
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})",
            line_1,
        )
        if not time_match:
            return None
        cur_text = f"{line_2} {line_3}".replace("&nbsp;", "").strip()
        return {
            "start_time": time_to_seconds(time_match.group(1)),
            "end_time": time_to_seconds(time_match.group(2)),
            "text": cur_text,
        }

    def _merge_segments_to_documents(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge continuous segments into documents based on chunk_size and chunk_overlap.

        Returns list of merged docs with 'start_time', 'end_time', 'text'.
        """
        if not segments:
            return []
        length_to = self._compute_cumulative_lengths(segments)
        start_idxs, ed_idxs = self._compute_chunk_boundaries(segments, length_to)
        if len(ed_idxs) < len(start_idxs):
            ed_idxs.append(len(segments) - 1)
        return [
            {
                "start_time": segments[s]["start_time"],
                "end_time": segments[e]["end_time"],
                "text": " ".join(segments[i]["text"] for i in range(s, e + 1)),
            }
            for s, e in zip(start_idxs, ed_idxs)
        ]

    def _compute_cumulative_lengths(self, segments: List[Dict[str, Any]]) -> List[int]:
        """Return cumulative character lengths: length_to[i] = sum of text lengths up to segment i."""
        length_to: List[int] = []
        for i, seg in enumerate(segments):
            cur_len = len(seg["text"])
            length_to.append(cur_len if i == 0 else length_to[i - 1] + cur_len)
        return length_to

    def _compute_chunk_boundaries(
        self, segments: List[Dict[str, Any]], length_to: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Compute start and end indices for chunks of ~chunk_size with chunk_overlap.
        Returns (start_idxs, ed_idxs).
        """
        start_idxs = [0]
        ed_idxs = []
        try:
            for i, segment in enumerate(segments):
                last_start = start_idxs[-1]
                len_before_i = length_to[i] - len(segment["text"])
                len_before_last_start = length_to[last_start] - len(
                    segments[last_start]["text"]
                )
                if (
                    len_before_i - len_before_last_start
                    > self.chunk_size - self.chunk_overlap
                ):
                    start_idxs.append(i)
                pair_idx = start_idxs[len(ed_idxs)]
                chunk_len = (
                    length_to[i] - length_to[pair_idx] + len(segments[pair_idx]["text"])
                )
                if chunk_len > self.chunk_size:
                    ed_idxs.append(i)
        except (IndexError, KeyError) as e:
            logger.error("Error computing chunk boundaries: %s", e)
            return [0], [len(segments) - 1]
        return start_idxs, ed_idxs

    def get_video_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available YouTube videos.

        Returns:
            Dictionary with statistics
        """
        self._load_video_ids()

        stats = {
            "total_videos": len(self.scripts_ids),
            "transcripts_available": len(self.scripts_ids),
            "metadata_available": len(self.metainfo_ids),
            "both_available": len(self.scripts_ids & self.metainfo_ids),
            "failed_videos": len(self.failed_video_ids),
        }

        return stats


def main():
    """Test the YouTube preprocessor."""
    preprocessor = YouTubePreprocessor()

    # Get statistics
    stats = preprocessor.get_video_statistics()
    print("YouTube Video Statistics:")
    print(json.dumps(stats, indent=2))

    # Load a few documents as test
    documents = preprocessor.load_documents(limit=5)
    print(f"\nLoaded {len(documents)} documents")

    if documents:
        print("\nSample document:")
        doc = documents[0]
        print(f"Channel: {doc.metadata.get('channel_title')}")
        print(f"Content preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
