"""Repeatable single-process benchmark. No reference means no accuracy claim."""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path
import sys
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--language", default="th")
parser.add_argument("--device", default="cuda")
parser.add_argument("--output", required=True)
parser.add_argument("--whisper-source")
parser.add_argument("--reference")
parser.add_argument("--beam-size", type=int)
parser.add_argument("--temperature")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--standard-initialization", action="store_true", help="Compare upstream parameter initialization on CPU")
args = parser.parse_args()
if args.whisper_source:
    sys.path.insert(0, str(Path(args.whisper_source).resolve()))
from yt_whisper.storage import configure_storage
storage = configure_storage()
import psutil
import torch
import whisper
import numpy as np
np.random.seed(args.seed)
torch.manual_seed(args.seed)
from yt_whisper.engine import Transcriber
from yt_whisper.results import save_result
from yt_whisper.audio import AudioSource

process = psutil.Process()
peak_rss = [process.memory_info().rss]
done = threading.Event()
def sample_memory():
    while not done.wait(.05):
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
monitor = threading.Thread(target=sample_memory, daemon=True)
monitor.start()
audio = Path(args.audio).resolve()
with audio.open("rb") as handle:
    audio_hash = hashlib.file_digest(handle, "sha256").hexdigest()
waveform = whisper.load_audio(str(audio))
duration = len(waveform) / 16000
del waveform
if args.device == "cuda":
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
engine = Transcriber(storage)
start = time.perf_counter()
if args.standard_initialization:
    engine._model = whisper.load_model(args.model, device=args.device, download_root=str(storage.models))
    engine._key = (args.model, args.device, None)
else:
    engine._load(args.model, args.device)
if args.device == "cuda":
    torch.cuda.synchronize()
load_seconds = time.perf_counter() - start
start = time.perf_counter()
result = engine.transcribe(audio, args.model, args.language, device=args.device, verbose=False,
                           beam_size=args.beam_size, temperature=args.temperature)
if args.device == "cuda":
    torch.cuda.synchronize()
infer_seconds = time.perf_counter() - start
done.set()
monitor.join()
metrics = {
    "model": args.model, "language": args.language, "device": args.device, "audio_seconds": duration,
    "audio_path": str(audio), "audio_sha256": audio_hash,
    "standard_initialization": args.standard_initialization, "seed": args.seed,
    "load_seconds": load_seconds, "transcribe_seconds": infer_seconds,
    "real_time_factor": infer_seconds / duration,
    "peak_sampled_rss_gib": peak_rss[0] / 2**30,
    "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/2**30 if args.device == "cuda" else None,
    "whisper_version": whisper.__version__, "whisper_source": whisper.__file__,
    "text_characters": len(result["text"]), "segments": len(result["segments"]),
    "reference_available": bool(args.reference),
    "decoding_options": result.get("decoding_options", {}),
}
if args.reference:
    # Character error rate; whitespace removed, no Thai word segmentation required.
    def chars(text):
        return "".join(text.split())
    reference_file = Path(args.reference).resolve()
    reference = chars(reference_file.read_text(encoding="utf-8"))
    metrics["reference_path"] = str(reference_file)
    metrics["reference_sha256"] = hashlib.sha256(reference_file.read_bytes()).hexdigest()
    metrics["reference_characters_no_whitespace"] = len(reference)
    hypothesis = chars(result["text"])
    if not reference:
        raise ValueError("Reference must not be empty")
    row = list(range(len(hypothesis)+1))
    for i, a in enumerate(reference, 1):
        new = [i]
        for j, b in enumerate(hypothesis, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(a!=b)))
        row = new
    metrics["character_error_rate"] = row[-1] / len(reference)
if args.whisper_source:
    metrics["whisper_commit"] = subprocess.check_output(
        ["git", "-C", args.whisper_source, "rev-parse", "HEAD"], text=True).strip()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
paths = save_result(result, AudioSource(audio, args.model, args.model, str(audio)),
                    Path(args.output).parent / "transcripts", ("txt", "json", "srt", "vtt", "tsv", "jsonl"))
metrics["outputs"] = paths
Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False), flush=True)
