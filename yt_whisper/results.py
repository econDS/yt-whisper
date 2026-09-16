import json
from pathlib import Path
import tempfile
from .utils import slugify, write_srt, write_vtt


def save_result(result, audio, output_dir, formats=("txt", "json", "srt", "vtt"), line_length=0):
    formats = tuple(formats)
    if not formats or set(formats) - {"txt", "json", "srt", "vtt", "tsv", "jsonl"}:
        raise ValueError("Select one or more supported output formats.")
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
            elif kind == "srt":
                write_srt(result["segments"], handle, line_length=line_length)
            elif kind == "vtt":
                write_vtt(result["segments"], handle, line_length=line_length)
            else:
                raise ValueError(f"Unsupported output format: {kind}")
        paths.append(str(path))
    return paths
