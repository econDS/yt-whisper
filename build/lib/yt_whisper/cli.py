import os
import whisper
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE
import argparse
import warnings
import yt_dlp
from .utils import str2bool, slugify, write_vtt, write_srt, convert_video_to_audio_ffmpeg
import tempfile
import torch


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="video URLs to transcribe")
    parser.add_argument("--model", default="base",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--format", default="vtt",
                        choices=["vtt", "srt"], help="the subtitle format to output")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="Whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, skip to perform language detection")

    parser.add_argument("--break-lines", type=int, default=0,
                        help="Whether to break lines into a bottom-heavy pyramid shape if line length exceeds N characters. 0 disables line breaking.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="If cuda selected pytorch will use GPU otherwise CPU")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    device: str = args.pop("device")
    output_dir: str = args.pop("output_dir")
    subtitles_format: str = args.pop("format")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"

    # check cuda availability
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, falling back to CPU. To use GPU, install PyTorch with CUDA support.")
        device = "cpu"

    model = whisper.load_model(model_name, device=device)
    audios = get_audio(args.pop("video"))
    break_lines = args.pop("break_lines")

    for title, audio_path in audios.items():
        warnings.filterwarnings("ignore")
        result = model.transcribe(audio_path, **args)
        warnings.filterwarnings("default")

        if subtitles_format == 'vtt':
            vtt_path = os.path.join(output_dir, f"{slugify(title)}.vtt")
            with open(vtt_path, 'w', encoding="utf-8") as vtt:
                write_vtt(result["segments"], file=vtt,
                          line_length=break_lines)

            print("Saved VTT to", os.path.abspath(vtt_path))
        elif subtitles_format == "srt":
            srt_path = os.path.join(output_dir, f"{slugify(title)}.srt")
            with open(srt_path, 'w', encoding="utf-8") as srt:
                write_srt(result["segments"], file=srt,
                          line_length=break_lines)
            print("Saved SRT to", os.path.abspath(srt_path))
        else:
            print(f"subtitle type {subtitles_format} is wrong")
            exit(-1)


def get_audio(urls):
    temp_dir = tempfile.gettempdir()

    ydl = yt_dlp.YoutubeDL({'format': 'bestaudio',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'verbose':False,
            "outtmpl": os.path.join(temp_dir, "%(id)s.%(ext)s"),
    })

    paths = {}
    for url in urls:
        if url.startswith('https://') or url.startswith("http://"):
            result = ydl.extract_info(url, download=True)
            print(
                f"Downloaded video \"{result['title']}\". Generating subtitles..."
            )
            paths[result["title"]] = os.path.join(
                temp_dir, f"{result['id']}.mp3")
        elif os.path.exists(url):
            print(f"local file {url}")
            output_mp3 = convert_video_to_audio_ffmpeg(url)
            paths[os.path.splitext(os.path.basename(url))[0]] = output_mp3
            assert os.path.exists(output_mp3)
            print(output_mp3)
        else:
            print(f"url not exist {url}")
    return paths


if __name__ == '__main__':
    main()
