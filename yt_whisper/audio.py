"""Downloaded sources own temporary directories, cleaned on failure too."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse


@dataclass(frozen=True)
class AudioSource:
    path: Path
    title: str
    source_id: str
    source: str


def require_ffmpeg():
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("FFmpeg is missing. Install it and add its bin folder to PATH.")
    return binary


def probe_audio_duration(audio):
    """Read the first audio stream's duration, falling back to container duration."""
    audio = Path(audio)
    if not audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio}")
    require_ffmpeg()
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFmpeg and ffprobe must be available on PATH.")
    try:
        run = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                              "format=duration:stream=codec_type,duration", "-of", "json", str(audio)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        data = json.loads(run.stdout)
        stream = next(s for s in data["streams"] if s.get("codec_type") == "audio")
        for value in (stream.get("duration"), data.get("format", {}).get("duration")):
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(duration) and duration > 0:
                return duration
        raise ValueError("Invalid audio duration")
    except (OSError, subprocess.CalledProcessError, KeyError, TypeError, ValueError, StopIteration) as exc:
        raise RuntimeError(f"Cannot read audio duration: {audio}") from exc


def load_audio_range(audio, start, end):
    """Decode one interval to Whisper's 16 kHz mono float32 format in memory."""
    import numpy as np
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("Audio ranges require finite times with 0 <= start < end.")
    # Input seeking with transcoding uses FFmpeg's default accurate seeking.
    # No full waveform or temporary clip is materialized on disk.
    command = [require_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", str(start), "-i", str(audio), "-t", str(end - start),
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    try:
        pcm = subprocess.run(command, capture_output=True, check=True).stdout
        waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        duration = len(waveform) / 16000.0
        if not duration or abs(duration - (end - start)) > .1:
            raise ValueError("Decoded audio duration does not match the requested range.")
        return waveform
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot decode audio range {start}–{end}: {exc}") from exc


def downloader_options(storage, directory):
    node = shutil.which("node")
    if not node:
        raise RuntimeError("YouTube requires Node.js >=22 in PATH (yt-dlp JavaScript runtime).")
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        raise RuntimeError(f"Node.js >=22 is required; found {version}.")
    return {
        "format": "bestaudio/best",
        "noplaylist": True,
        "outtmpl": "%(id)s.%(ext)s",
        "paths": {"home": str(directory), "temp": str(directory)},
        "cachedir": str(storage.root / "cache/yt-dlp"),
        "js_runtimes": {"node": {"path": node}},
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }


@contextmanager
def prepare_audio(source, storage):
    require_ffmpeg()
    source = str(source).strip()
    if not source:
        raise ValueError("Choose an audio/video file or enter a URL.")
    if urlparse(source).scheme.lower() not in ("http", "https"):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio/video file does not exist: {path}")
        # Whisper/FFmpeg decode directly; never convert beside the input.
        yield AudioSource(path, path.stem, path.stem, str(path))
        return
    import yt_dlp
    with tempfile.TemporaryDirectory(prefix="download-", dir=storage.temp) as directory:
        options = downloader_options(storage, directory)
        options["extract_flat"] = "in_playlist"
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=False)
            if not info or info.get("_type") in ("playlist", "multi_video") or "entries" in info:
                raise ValueError("Playlists are not supported. Enter a single video URL.")
            info = downloader.process_ie_result(info, download=True)
        audio = Path(directory) / f"{info['id']}.wav"
        if not audio.is_file():
            raise RuntimeError("The downloader did not produce an audio file.")
        yield AudioSource(audio, info.get("title") or info["id"], str(info["id"]), source)
