"""Local Gradio interface. Importing this module does not launch a server."""
from .storage import configure_storage


def build_demo():
    storage = configure_storage()
    import gradio as gr
    import whisper
    from whisper.tokenizer import LANGUAGES
    from .audio import prepare_audio
    from .config import load_config, save_config
    from .engine import THAI_MODEL, Transcriber, normalize_language
    from .options import decoding_options
    from .results import save_result

    config = load_config(storage)
    models = whisper.available_models() + [THAI_MODEL]
    selected = config["model"] if config["model"] in models else "base"
    try:
        language = normalize_language(config["language"]) or "Auto"
    except ValueError:
        language = "Auto"
    engine = Transcriber(storage)

    def transcribe(source, model, language, task, device, initial_prompt="", word_timestamps=False,
                   carry_initial_prompt=False, condition_on_previous_text=True,
                   silence_threshold=None, clip_timestamps="0", beam_size=None, temperature="",
                   output_formats=None):
        if not source:
            raise gr.Error("Choose a file or enter a video URL.")
        try:
            options = {} if model == THAI_MODEL else dict(
                initial_prompt=initial_prompt, word_timestamps=word_timestamps,
                carry_initial_prompt=carry_initial_prompt,
                condition_on_previous_text=condition_on_previous_text,
                hallucination_silence_threshold=silence_threshold,
                clip_timestamps=clip_timestamps, beam_size=beam_size, temperature=temperature,
            )
            options = decoding_options(model, **options)
            formats = ["txt", "json", "srt", "vtt"] if output_formats is None else output_formats
            if not formats:
                raise ValueError("Select at least one output format.")
            with prepare_audio(source, storage) as audio:
                result = engine.transcribe(audio.path, model, language, task, device, **dict(options, verbose=False))
                files = save_result(result, audio, storage.outputs, formats)
            save_config(storage, model, language)
            return result["text"].strip(), files
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="YouTube & File Whisper", delete_cache=(3600, 86400)) as demo:
        gr.Markdown("# Whisper transcription\nTranscribe a local audio/video file or a single video URL.")
        with gr.Row():
            model = gr.Dropdown(models, value=selected, label="Model")
            language_input = gr.Dropdown(
                [("Auto detect", "Auto")] + [(name.title(), code) for code, name in sorted(LANGUAGES.items(), key=lambda p: p[1])],
                value=language, label="Spoken language",
            )
            task = gr.Dropdown([("Transcribe", "transcribe"), ("Translate to English", "translate")],
                               value="transcribe", label="Task")
            device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
        gr.Markdown("Base is useful for quick tests. Turbo is faster but cannot translate to English. "
                    "Large-v3 needs more memory. Thonburian uses its own Thai transcription settings.")
        with gr.Accordion("Transcription options — OpenAI Whisper", open=False,
                          visible=selected != THAI_MODEL) as advanced:
            prompt = gr.Textbox(label="Names, terms and context", placeholder="Spellings or vocabulary to help recognition")
            with gr.Row():
                words = gr.Checkbox(label="Word timestamps", value=False)
                carry = gr.Checkbox(label="Repeat prompt for each window", value=False)
                previous = gr.Checkbox(label="Use previous text as context", value=True,
                                      info="Try turning off if phrases repeat.")
            with gr.Row():
                silence = gr.Number(label="Skip suspected hallucinations after silence (seconds)", value=None,
                                    info="Requires word timestamps; leave empty to disable.")
                clips = gr.Textbox(label="Audio ranges (seconds)", value="0",
                                   info="start,end,start,end,...; a final start runs to the end.")
            with gr.Row():
                beam = gr.Number(label="Beam size", value=None, precision=0,
                                 info="Positive integer; used only at temperature 0.")
                temperature = gr.Textbox(label="Temperature / fallback sequence", value="",
                                         info="0–1, e.g. 0,0.2,0.4. Empty uses Whisper defaults.")
        model.change(lambda value: gr.update(visible=value != THAI_MODEL), model, advanced, api_name=False)
        formats = gr.CheckboxGroup(["txt", "json", "srt", "vtt", "tsv", "jsonl"],
                                   value=["txt", "json", "srt", "vtt"], label="Output formats")
        with gr.Tab("URL"):
            url = gr.Textbox(label="Video URL", placeholder="https://www.youtube.com/watch?v=...")
            url_button = gr.Button("Transcribe URL", variant="primary")
        with gr.Tab("File"):
            upload = gr.File(label="Audio or video file", type="filepath")
            file_button = gr.Button("Transcribe file", variant="primary")
        output = gr.Textbox(label="Transcription", lines=12)
        downloads = gr.File(label="Download transcripts", file_count="multiple", interactive=False)
        settings = [model, language_input, task, device, prompt, words, carry, previous, silence, clips, beam, temperature, formats]
        for button, source, api in ((url_button, url, "transcribe_url"), (file_button, upload, "transcribe_file")):
            button.click(transcribe, [source] + settings, [output, downloads],
                         concurrency_limit=1, concurrency_id="transcription", api_name=api)
    return demo


def main():
    storage = configure_storage()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False, allowed_paths=[str(storage.outputs)],
    )


if __name__ == "__main__":
    main()
