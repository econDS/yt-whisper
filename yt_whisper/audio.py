"""Downloaded sources own temporary directories, cleaned on failure too."""
from contextlib import contextmanager
from dataclasses import dataclass
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
