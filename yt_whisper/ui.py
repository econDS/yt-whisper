"""Local Gradio interface. Importing this module does not launch a server."""
import math
from pathlib import Path
import time
from .storage import configure_storage


def time_range_timestamps(start, end, multiple="0"):
    """Combine the convenient single-range controls with the existing range option."""
    from .options import parse_clip_timestamps, timestamp_seconds
    start, end = (str(value or "").strip() for value in (start, end))
    if not start and not end:
        return multiple
    if parse_clip_timestamps(multiple) != "0":
        raise ValueError("Use either Start / End time or Multiple audio ranges; clear the other fields.")
    start_seconds = timestamp_seconds(start or "0")
    if not end:
        return parse_clip_timestamps([start_seconds])
    end_seconds = timestamp_seconds(end)
    if end_seconds <= start_seconds:
        raise ValueError("End time must be later than Start time.")
    return [start_seconds, end_seconds]


def output_directory(storage, subfolder):
    root = storage.outputs.resolve()
    relative = Path(subfolder or "")
    destination = (root / relative).resolve()
    if relative.is_absolute() or not destination.is_relative_to(root):
        raise ValueError("Choose an output subfolder inside the configured outputs folder.")
    return destination


def build_demo():
    storage = configure_storage()
    import gradio as gr
    from whisper.tokenizer import LANGUAGES
    from .audio import prepare_audio
    from .config import load_config, save_config
    from .engine import Transcriber, normalize_language, validate_selection
    from .models import model_choices, supports_openai_options
    from .options import decoding_options, optional_float
    from .results import save_result, subtitle_options

    config = load_config(storage)
    models = model_choices()
    selected = config["model"] if config["model"] in dict(models).values() else "base"
    try:
        language = normalize_language(config["language"]) or "Auto"
    except ValueError:
        language = "Auto"
    engine = Transcriber(storage)

    def transcribe(source, model, language, task, device, initial_prompt="", word_timestamps=False,
                   carry_initial_prompt=False, condition_on_previous_text=True,
                   silence_threshold=None, clip_timestamps="0", beam_size=None, temperature="",
                   output_formats=None, preset="default", best_of=None,
                   compression_ratio_threshold="", logprob_threshold="", no_speech_threshold="",
                   highlight_words=False, max_line_width=None, max_line_count=None,
                   max_words_per_line=None, break_lines=0, output_subfolder="",
                   start_time="", end_time="", progress=gr.Progress()):
        if not source:
            raise gr.Error("Choose a file or enter a video URL.")
        try:
            if (isinstance(break_lines, bool) or not isinstance(break_lines, (int, float))
                    or not math.isfinite(break_lines) or break_lines < 0 or int(break_lines) != break_lines):
                raise ValueError("Subtitle line length must be a non-negative integer; 0 disables wrapping.")
            destination = output_directory(storage, output_subfolder)
            model, language = validate_selection(model, language, task)
            openai_options = supports_openai_options(model)
            clip_timestamps = time_range_timestamps(start_time, end_time, clip_timestamps)
            options = {"clip_timestamps": clip_timestamps} if not openai_options else dict(
                initial_prompt=initial_prompt, word_timestamps=word_timestamps,
                carry_initial_prompt=carry_initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                hallucination_silence_threshold=silence_threshold,
                clip_timestamps=clip_timestamps, beam_size=beam_size, temperature=temperature,
            )
            if openai_options:
                # Empty UI fields inherit the preset; "none" explicitly disables thresholds.
                options["preset"] = preset
                if beam_size is None:
                    options.pop("beam_size")
                if temperature in ("", None):
                    options.pop("temperature")
                if best_of is not None:
                    options["best_of"] = best_of
                for key, value in (
                    ("compression_ratio_threshold", compression_ratio_threshold),
                    ("logprob_threshold", logprob_threshold),
                    ("no_speech_threshold", no_speech_threshold),
                ):
                    if value is not None and str(value).strip():
                        options[key] = optional_float(value)
            options = decoding_options(model, **options)
            subtitles = {} if not openai_options else subtitle_options(
                word_timestamps, highlight_words, max_line_width, max_line_count, max_words_per_line,
                line_length=break_lines)
            formats = ["txt", "json", "srt", "vtt"] if output_formats is None else output_formats
            if not formats:
                raise ValueError("Select at least one output format.")
            progress(0, desc="Preparing audio / downloading video")
            with prepare_audio(source, storage) as audio:
                progress(.15, desc="Loading model and transcribing")
                result = engine.transcribe(audio.path, model, language, task, device, **dict(options, verbose=False))
                progress(.9, desc="Saving transcripts")
                files = save_result(result, audio, destination, formats, int(break_lines), subtitles=subtitles)
            save_config(storage, model, language)
            progress(1, desc="Complete")
            return result["text"].strip(), files
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    def transcribe_interactive(*args, progress=gr.Progress()):
        if len(args) == 26:  # Existing UI callers may omit the new time fields.
            args += ("", "")
        request = args[:-5]
        mode, line_length, subfolder, start_time, end_time = args[-5:]
        began = time.perf_counter()
        yield "", [], "Processing..."
        try:
            if mode == "Simple":
                # Only the visible model, language and time range apply in Simple mode.
                text, files = transcribe(*request[:3], "transcribe", "auto",
                                         start_time=start_time, end_time=end_time, progress=progress)
            elif mode == "Advanced":
                text, files = transcribe(*request, break_lines=line_length,
                                         output_subfolder=subfolder, start_time=start_time,
                                         end_time=end_time, progress=progress)
            else:
                raise ValueError("Choose Simple or Advanced mode.")
            elapsed = time.perf_counter() - began
            yield text, files, f"Completed in {elapsed:.2f} seconds | {len(files)} files saved"
        except Exception as exc:
            elapsed = time.perf_counter() - began
            yield "", [], f"Failed after {elapsed:.2f} seconds: {exc}"

    with gr.Blocks(title="YouTube & File Whisper", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# Whisper\nA file or video URL. A transcript you can copy or download.")
        mode = gr.Radio(["Simple", "Advanced"], value="Simple", label="Mode",
                        info="Simple: transcribe with automatic device selection and default settings.")
        with gr.Row():
            # Accept legacy API values; validate_selection rejects unknown names.
            model = gr.Dropdown(models, value=selected, label="Model", allow_custom_value=True)
            language_input = gr.Dropdown(
                [("Auto detect", "Auto")] + [(name.title(), code) for code, name in sorted(LANGUAGES.items(), key=lambda p: p[1])],
                value=language, label="Spoken language",
            )
        with gr.Accordion("Time range (optional)", open=False):
            with gr.Row():
                start_time = gr.Textbox(label="Start time", value="", placeholder="e.g. 10:00",
                                        info="MM:SS, HH:MM:SS or seconds. Empty starts at the beginning.")
                end_time = gr.Textbox(label="End time", value="", placeholder="e.g. 12:30",
                                      info="Empty continues to the end. Leave both empty for the whole recording.")
            gr.Markdown("Transcribes only this interval. Video URLs still download the full audio. "
                        "Subtitle times refer to the original recording. "
                        "Cuts through speech can produce repeated text; review the transcript.")
            clips = gr.Textbox(label="Multiple audio ranges", value="0", visible=False,
                               info="e.g. 10:00,12:30,30:00,31:00; seconds also work. "
                                    "Leave Start / End time empty when using this field. A final start runs to the end.")
        with gr.Tab("File"):
            upload = gr.File(label="Audio or video file", type="filepath")
            file_button = gr.Button("Transcribe file", variant="primary")
        with gr.Tab("URL"):
            url = gr.Textbox(label="Video URL", placeholder="https://www.youtube.com/watch?v=...")
            url_button = gr.Button("Transcribe URL", variant="primary")
        with gr.Row(visible=False) as task_settings:
            task = gr.Dropdown([("Transcribe", "transcribe"), ("Translate to English", "translate")],
                               value="transcribe", label="Task")
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        model_guide = gr.Markdown("Base is useful for quick tests. Turbo is faster but cannot translate to English. "
                    "Thonburian models are Thai-specialized and support transcription only: "
                    "Medium is the existing baseline, Large-v3 is the full larger model, "
                    "and Distilled Large-v3 is a smaller alternative. Select Auto or Thai; "
                    "compare speed, memory and accuracy on your own audio.", visible=False)
        with gr.Accordion("Transcription options — OpenAI Whisper", open=False,
                          visible=False) as advanced:
            preset = gr.Dropdown(
                [("Current defaults", "default"), ("Official Whisper CLI decoding", "official-cli")],
                value="default", label="Decoding preset",
                info="Official: beam 5, best-of 5, temperature 0,0.2,0.4,0.6,0.8,1. Filled fields override it.")
            prompt = gr.Textbox(label="Names, terms and context", placeholder="Spellings or vocabulary to help recognition")
            with gr.Row():
                words = gr.Checkbox(label="Word timestamps", value=False)
                carry = gr.Checkbox(label="Repeat prompt for each window", value=False)
                previous = gr.Checkbox(label="Use previous text as context", value=True,
                                      info="Try turning off if phrases repeat.")
            with gr.Row():
                silence = gr.Number(label="Skip suspected hallucinations after silence (seconds)", value=None,
                                    info="Requires word timestamps; leave empty to disable.")
            with gr.Row():
                beam = gr.Number(label="Beam size", value=None, precision=0,
                                 info="Positive integer; used only at temperature 0.")
                temperature = gr.Textbox(label="Temperature / fallback sequence", value="",
                                         info="0–1, e.g. 0,0.2,0.4. Empty inherits the preset; 0 disables fallback.")
            with gr.Row():
                best_of = gr.Number(label="Best of", value=None, precision=0,
                                    info="Candidates at nonzero temperatures; empty inherits the preset.")
                compression = gr.Textbox(label="Compression ratio threshold", value="",
                                         info="Empty: 2.4. Type none to disable repetition detection.")
                logprob = gr.Textbox(label="Log probability threshold", value="",
                                     info="Empty: -1.0. Type none to disable low-confidence fallback.")
                no_speech = gr.Textbox(label="No speech threshold", value="",
                                       info="Empty: 0.6. Type none to disable silence skipping.")
            with gr.Accordion("Subtitle layout (SRT/VTT)", open=False):
                gr.Markdown("Enable word timestamps above. Layout affects subtitle files only.")
                highlight = gr.Checkbox(label="Highlight each spoken word", value=False)
                with gr.Row():
                    line_width = gr.Number(label="Maximum characters per line", value=None, precision=0)
                    line_count = gr.Number(label="Maximum lines per cue", value=None, precision=0,
                                           info="Requires maximum characters per line.")
                    words_per_line = gr.Number(label="Maximum words per line", value=None, precision=0,
                                               info="Use instead of maximum characters per line.")
        with gr.Accordion("Export settings", open=False, visible=False) as export_settings:
            formats = gr.CheckboxGroup(["txt", "json", "srt", "vtt", "tsv", "jsonl"],
                                   value=["txt", "json", "srt", "vtt"], label="Output formats")
            break_lines = gr.Number(label="Segment subtitle line length", value=0, precision=0,
                                    info="CLI --break-lines: 0 disables wrapping. Use instead of word-based layout.")
            subfolder = gr.Textbox(label="Output subfolder", value="", placeholder="e.g. interviews/session-1",
                                  info=f"Saved under {storage.outputs}. Empty uses the main output folder.")
        with gr.Accordion("System info", open=False, visible=False) as system_info:
            check_setup = gr.Button("Check setup", size="sm")
            setup_report = gr.JSON(label="Paths, dependencies and GPU")
            from .cli import doctor
            check_setup.click(lambda: doctor(storage), outputs=setup_report, api_name=False)
        def show_mode(value, selected_model):
            enabled = value == "Advanced"
            return (gr.update(visible=enabled), gr.update(visible=enabled),
                    gr.update(visible=enabled and supports_openai_options(selected_model)),
                    gr.update(visible=enabled), gr.update(visible=enabled), gr.update(visible=enabled))
        visibility_outputs = [task_settings, model_guide, advanced, export_settings, system_info, clips]
        mode.change(show_mode, [mode, model], visibility_outputs, api_name=False, queue=False)
        model.change(show_mode, [mode, model], visibility_outputs, api_name=False, queue=False)
        status = gr.Markdown("Ready")
        gr.Markdown("<small>Processing time includes model loading; excludes browser upload and queue wait.</small>")
        output = gr.Textbox(label="Transcription", lines=10, buttons=["copy"], interactive=True,
                            placeholder="Your transcript will appear here. You can edit it before copying.",
                            info="Copy using the toolbar button. Edits affect copied text; downloads keep the original output.")
        downloads = gr.File(label="Download transcripts", file_count="multiple", interactive=False)
        settings = [model, language_input, task, device, prompt, words, carry, previous, silence, clips, beam, temperature, formats,
                    preset, best_of, compression, logprob, no_speech, highlight, line_width, line_count, words_per_line]
        for button, source, api in ((url_button, url, "transcribe_url"), (file_button, upload, "transcribe_file")):
            # Preserve the two-output API used by existing gradio_client callers.
            legacy = gr.Button(visible=False)
            legacy.click(transcribe, [source] + settings, [output, downloads],
                         concurrency_limit=1, concurrency_id="transcription", api_name=api)
            button.click(transcribe_interactive, [source] + settings + [mode, break_lines, subfolder, start_time, end_time],
                         [output, downloads, status], show_progress="full",
                         concurrency_limit=1, concurrency_id="transcription", api_name=api + "_ui")
    return demo


def main():
    storage = configure_storage()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, allowed_paths=[str(storage.outputs)],
    )


if __name__ == "__main__":
    main()
