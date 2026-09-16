import json
import math
from pathlib import Path
import tempfile
from .utils import slugify, write_srt, write_vtt


def subtitle_options(word_timestamps, highlight_words=False, max_line_width=None,
                     max_line_count=None, max_words_per_line=None, *, line_length=0):
    """Validate layout before expensive transcription or creating output files."""
    if type(highlight_words) is not bool:
        raise ValueError("highlight_words must be a boolean.")
    values = dict(highlight_words=highlight_words, max_line_width=max_line_width,
                  max_line_count=max_line_count, max_words_per_line=max_words_per_line)
    for key in ("max_line_width", "max_line_count", "max_words_per_line"):
        value = values[key]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not value >= 1 or value != int(value):
                raise ValueError(f"{key} must be a positive integer.")
            values[key] = int(value)
    if not any(values.values()):
        return {}
    if not word_timestamps:
        raise ValueError("Subtitle layout/highlighting requires word timestamps.")
    if line_length:
        raise ValueError("Use either --break-lines or word-based subtitle layout.")
    if max_line_count is not None and max_line_width is None:
        raise ValueError("max_line_count requires max_line_width.")
    if max_words_per_line is not None and max_line_width is not None:
        raise ValueError("Choose max_words_per_line or max_line_width, not both.")
    return values


def save_result(result, audio, output_dir, formats=("txt", "json", "srt", "vtt"), line_length=0, *, subtitles=None):
    formats = tuple(formats)
    if not formats or set(formats) - {"txt", "json", "srt", "vtt", "tsv", "jsonl"}:
        raise ValueError("Select one or more supported output formats.")
    subtitles = subtitle_options(True, **(subtitles or {}), line_length=line_length)
    if subtitles and any("words" not in segment for segment in result["segments"]):
        raise ValueError("Word timestamps are missing from the transcription.")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    prefix = (slugify(audio.title)[:60] or "transcript") + "-"
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    payload = dict(result, source=audio.source, source_id=audio.source_id, title=audio.title)
    paths = []
    for kind in formats:
        path = directory / f"transcript.{kind}"
        with path.open("w", encoding="utf-8") as handle:
            if kind == "txt":
                handle.write(result["text"].strip() + "\n")
            elif kind == "json":
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            elif kind == "jsonl":
                for segment in result["segments"]:
                    handle.write(json.dumps(segment, ensure_ascii=False) + "\n")
            elif kind == "tsv":
                handle.write("start\tend\ttext\n")
                for segment in result["segments"]:
                    text = segment["text"].strip().replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    handle.write(f"{round(segment['start'] * 1000)}\t{round(segment['end'] * 1000)}\t{text}\n")
            elif kind in ("srt", "vtt") and subtitles:
                from whisper.utils import WriteSRT, WriteVTT
                writer = (WriteSRT if kind == "srt" else WriteVTT)(str(directory))
                writer.write_result(result, handle, options=subtitles)
            elif kind == "srt":
                write_srt(result["segments"], handle, line_length=line_length)
            elif kind == "vtt":
                write_vtt(result["segments"], handle, line_length=line_length)
            else:
                raise ValueError(f"Unsupported output format: {kind}")
        paths.append(str(path))
    return paths
