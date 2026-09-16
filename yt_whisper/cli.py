import argparse
import importlib.metadata
import json
import shutil
import sys
from .storage import configure_storage


def doctor(storage):
    import torch
    report = {
        "python": sys.version, "executable": sys.executable,
        "storage": str(storage.root), "temp": str(storage.temp),
        "whisper_models": str(storage.models), "thai_model": str(storage.thai_model),
        "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"), "node": shutil.which("node"),
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": {}, "cached_models": [p.name for p in storage.models.glob("*.pt")],
    }
    for name in ("openai-whisper", "torch", "gradio", "transformers", "yt-dlp", "yt-dlp-ejs"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    return report


def main(argv=None):
    storage = configure_storage()
    import whisper
    from yt_dlp.utils import DownloadError
    from .audio import prepare_audio
    from .engine import THAI_MODEL, Transcriber
    from .results import save_result, subtitle_options
    from .options import decoding_options, optional_float, PRESETS
    from .utils import str2bool

    parser = argparse.ArgumentParser(description="Transcribe audio/video files and single YouTube videos.")
    parser.add_argument("video", nargs="*", help="Local files or video URLs")
    parser.add_argument("--doctor", action="store_true", help="Report paths/dependencies without loading models")
    parser.add_argument("--model", default="base", choices=whisper.available_models() + [THAI_MODEL])
    parser.add_argument("--format", default="vtt", choices=["txt", "json", "srt", "vtt", "tsv", "jsonl", "all"])
    parser.add_argument("--output_dir", "-o", default=str(storage.outputs))
    parser.add_argument("--language", default=None, help="Language name/code, or Auto")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--break-lines", type=int, default=0)
    parser.add_argument("--initial-prompt", default=None)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--carry-initial-prompt", action="store_true")
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hallucination-silence-threshold", type=float, default=None)
    parser.add_argument("--clip-timestamps", default="0", help="Seconds: start,end,start,end,...")
    parser.add_argument("--beam-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", default=argparse.SUPPRESS, help="0..1 or comma-separated fallback sequence")

    parser.add_argument("--preset", choices=list(PRESETS), default="default",
                        help="official-cli uses beam=5, best_of=5 and temperature fallback; explicit options override it")
    parser.add_argument("--best-of", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--compression-ratio-threshold", type=optional_float, default=argparse.SUPPRESS)
    parser.add_argument("--logprob-threshold", type=optional_float, default=argparse.SUPPRESS)
    parser.add_argument("--no-speech-threshold", type=optional_float, default=argparse.SUPPRESS,
                        help="0..1, or none to disable")
    parser.add_argument("--highlight-words", action="store_true")
    parser.add_argument("--max-line-width", type=int)
    parser.add_argument("--max-line-count", type=int)
    parser.add_argument("--max-words-per-line", type=int)
    args = parser.parse_args(argv)
    if args.doctor:
        print(json.dumps(doctor(storage), ensure_ascii=False, indent=2))
        return 0
    if not args.video:
        parser.error("Provide at least one file/URL, or use --doctor.")
    engine = Transcriber(storage)
    formats = ("txt", "json", "srt", "vtt", "tsv", "jsonl") if args.format == "all" else (args.format,)
    try:
        options = decoding_options(
            args.model, verbose=args.verbose, initial_prompt=args.initial_prompt,
            word_timestamps=args.word_timestamps, carry_initial_prompt=args.carry_initial_prompt,
            condition_on_previous_text=args.condition_on_previous_text,
            hallucination_silence_threshold=args.hallucination_silence_threshold,
            clip_timestamps=args.clip_timestamps, preset=args.preset,
            **{key: getattr(args, key) for key in (
                "beam_size", "best_of", "temperature", "compression_ratio_threshold",
                "logprob_threshold", "no_speech_threshold") if hasattr(args, key)},
        )
        subtitles = subtitle_options(options.get("word_timestamps", False),
            args.highlight_words, args.max_line_width, args.max_line_count, args.max_words_per_line,
            line_length=args.break_lines)
        for source in args.video:
            with prepare_audio(source, storage) as audio:
                result = engine.transcribe(audio.path, args.model, args.language, args.task, args.device,
                                           **options)
                for path in save_result(result, audio, args.output_dir, formats, args.break_lines, subtitles=subtitles):
                    print(f"Saved: {path}")
    except (RuntimeError, ValueError, OSError, DownloadError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
