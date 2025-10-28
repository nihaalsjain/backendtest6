import os
import re
import io
import time
import base64
import logging
import subprocess
import urllib.request
from pathlib import Path

import yt_dlp
import webvtt
import translators as ts
from moviepy import VideoFileClip
from dotenv import load_dotenv
import openai

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# -------------------------
# Load ENV
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DB_DIR = Path("vectorstore_multi_pdf")
VIDEO_CACHE_DIR = Path("video_cache")
VIDEO_CACHE_DIR.mkdir(exist_ok=True)

# -------------------------
# VIDEO HELPERS
# -------------------------

def download_youtube_video(url, out_dir):
    """Download a YouTube video (or use cache)."""
    out_dir = Path(out_dir)
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info["id"]
        expected_path = out_dir / f"{video_id}.mp4"

    if expected_path.exists():
        logger.info(f"✅ Using cached video: {expected_path}")
        return expected_path, info

    logger.info(f"⬇️ Downloading video from {url}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    logger.info("Video download complete.")
    return expected_path, info


def extract_audio(video_path, audio_path):
    """Extract audio from video → .mp3 (MoviePy → fallback FFmpeg)."""
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    logger.info(f"Extracting audio from {video_path}...")

    try:
        with VideoFileClip(str(video_path)) as clip:
            if clip.audio:
                clip.audio.write_audiofile(str(audio_path))
                logger.info(f"Audio extracted with MoviePy → {audio_path}")
                return True
            else:
                logger.warning("No audio track found in video.")
                return False
    except Exception as e:
        logger.warning(f"MoviePy failed: {e}. Trying FFmpeg...")

    try:
        command = ["ffmpeg", "-i", str(video_path), "-q:a", "0", "-map", "a", "-vn", str(audio_path), "-y"]
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Audio extracted with FFmpeg → {audio_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        return False


def transcribe_audio_with_whisper(audio_path):
    """Use OpenAI Whisper API to transcribe audio."""
    logger.info(f"🎙️ Transcribing audio with Whisper → {audio_path}")
    if not OPENAI_API_KEY:
        logger.error("❌ No OpenAI API key.")
        return []

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
            )
        raw_segments = response.segments if hasattr(response, "segments") else response.get("segments", [])
        if not raw_segments:
            return []

        transcript = []
        for seg in raw_segments:
            if isinstance(seg, dict):
                start, end, text = seg.get("start"), seg.get("end"), seg.get("text", "").strip()
            else:
                start, end, text = seg.start, seg.end, seg.text.strip()
            transcript.append({"start": start, "end": end, "text": text})

        logger.info(f"✅ Transcription returned {len(transcript)} segments")
        return transcript
    except Exception as e:
        logger.error(f"Whisper API failed: {e}", exc_info=True)
        return []


def get_transcript(url, video_path):
    """Tiered: try captions first, then Whisper transcription."""
    logger.info(f"Fetching transcript for {url}...")
    ydl_opts = {"writeautomaticsub": True, "writesubtitles": True, "skip_download": True, "subtitleslangs": ["en"], "quiet": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get("subtitles") or info.get("automatic_captions")
            if subs:
                subtitle_url = subs.get("en", [])[0]["url"] if "en" in subs else list(subs.values())[0][0]["url"]
                with urllib.request.urlopen(subtitle_url) as response:
                    vtt_content = response.read().decode("utf-8")
                    captions = webvtt.read_buffer(io.StringIO(vtt_content))
                    transcript = []
                    for cap in captions:
                        text = cap.text.replace("\n", " ").strip()
                        if "en" not in subs:
                            text = ts.translate_text(text, translator="google", to_language="en")
                        transcript.append({"text": text, "start": cap.start_in_seconds, "end": cap.end_in_seconds})
                    if transcript:
                        return transcript
    except Exception as e:
        logger.error(f"Subtitle processing failed: {e}")

    logger.warning("⚠️ No usable subtitles. Falling back to Whisper.")
    audio_path = video_path.with_suffix(".mp3")
    if not extract_audio(video_path, audio_path):
        return []
    return transcribe_audio_with_whisper(audio_path)


def sample_frames(video_path, out_dir, every_seconds=5):
    """Sample frames with ffmpeg (1 per N seconds)."""
    frames_dir = Path(out_dir) / "frames"
    frames_dir.mkdir(exist_ok=True)
    cmd = ["ffmpeg", "-i", str(video_path), "-vf", f"fps=1/{every_seconds}", f"{frames_dir}/frame_%d.jpg", "-loglevel", "error"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Frame sampling failed: {e}")
        return []
    return sorted(frames_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split("_")[1]))


def _read_and_b64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_video_documents(transcript_data, frame_paths, frame_interval_seconds):
    """Combine transcript + frames into structured docs."""
    if not frame_paths:
        return []
    frame_map = {i * frame_interval_seconds: path for i, path in enumerate(frame_paths, 1)}
    docs = []
    for seg in transcript_data:
        seg_start = seg.get("start", 0.0)
        closest_time = min(frame_map.keys(), key=lambda t: abs(t - seg_start))
        docs.append({
            "text": seg["text"],
            "start": seg.get("start"),
            "end": seg.get("end"),
            "frame_path": frame_map[closest_time],
            "timestamp": closest_time,
        })
    return docs


def build_captions_for_specific_docs(video_docs, llm_vision: ChatOpenAI):
    """Generate captions for selected frames only (rate-limit safe)."""
    enriched_docs = []
    for doc in video_docs:
        caption = "No caption available."
        try:
            b64_image = _read_and_b64(doc["frame_path"])
            msg = HumanMessage(content=[
                {"type": "text", "text": "Describe this image from a vehicle repair video. Focus on tools, parts, actions."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ])
            response = llm_vision.invoke([msg])
            caption = response.content
            time.sleep(1)
        except Exception as e:
            logger.error(f"Caption generation failed: {e}")

        enriched_docs.append({
            "combined_text": f"Transcript: \"{doc['text']}\"\nVisual Context: {caption}",
            "start": doc["start"], "end": doc["end"], "frame_path": doc["frame_path"], "timestamp": doc["timestamp"],
        })
    return enriched_docs


def retrieve_relevant_video_segments(documents, query, embeddings_model, top_k, video_id):
    """Search Chroma for best-matching video docs."""
    collection_name = f"vid_{video_id}"
    texts = [doc["combined_text"] for doc in documents]
    client = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings_model, collection_name=collection_name)

    if client._collection.count() == 0:
        logger.info(f"Indexing {len(texts)} docs in '{collection_name}'...")
        client.add_texts(texts=texts, metadatas=[{"source": f"doc_{i}"} for i in range(len(texts))])

    try:
        results = client.similarity_search(query, k=top_k)
        result_texts = [doc.page_content for doc in results]
        return [doc for doc in documents if doc["combined_text"] in result_texts]
    except Exception as e:
        logger.error(f"Vectorstore retrieval failed: {e}")
        return []


def process_video_query(youtube_url, prompt, llm_vision: ChatOpenAI, embeddings: OpenAIEmbeddings, top_k=5) -> str:
    """Run the full RAG + vision pipeline for a YouTube video."""
    try:
        video_path, info = download_youtube_video(youtube_url, out_dir=VIDEO_CACHE_DIR)
        video_title = info.get("title", "YouTube Video")
        video_url_base = info.get("webpage_url", youtube_url)
        video_id = info.get("id")
        frame_interval = 5

        transcript_data = get_transcript(youtube_url, video_path)
        frame_paths = sample_frames(video_path, VIDEO_CACHE_DIR, frame_interval)

        if not transcript_data or not frame_paths:
            return f"Could not analyze '{video_title}'. Transcript or frames unavailable."

        docs = create_video_documents(transcript_data, frame_paths, frame_interval)
        transcript_only_docs = [{"combined_text": d["text"], "original_doc": d} for d in docs]

        candidates = retrieve_relevant_video_segments(
            documents=transcript_only_docs,
            query=prompt,
            embeddings_model=embeddings,
            top_k=top_k,
            video_id=f"{video_id}_transcript_only",
        )
        top_docs = [c["original_doc"] for c in candidates] or docs[:top_k]

        enriched = build_captions_for_specific_docs(top_docs, llm_vision)
        if not enriched:
            return "No relevant moments found."

        doc = enriched[0]
        b64_image = _read_and_b64(doc["frame_path"])
        vision_prompt = f"""You are a precise video analysis assistant for automotive repair.
        USER'S QUESTION: "{prompt}"
        VIDEO CONTEXT: "{doc['combined_text']}"
        Provide a structured report:
        ## Problem Diagnosis
        [describe based only on video]
        ## Step-by-Step Solution
        [numbered list with (timestamp: XXs)]"""

        msg = HumanMessage(content=[
            {"type": "text", "text": vision_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
        ])
        summary = llm_vision.invoke([msg]).content

        return f"**Video Title:** {video_title}\n\n---\n{summary}"
    except Exception as e:
        logger.error(f"Video query failed: {e}", exc_info=True)
        return f"❌ Error analyzing video: {e}"
