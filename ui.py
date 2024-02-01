import gradio as gr
import json
import os
import torch
import whisper
from transformers import pipeline
from user_config import user_config
from whisper.tokenizer import TO_LANGUAGE_CODE
from yt_whisper.cli import get_audio

os.environ['KMP_DUPLICATE_LIB_OK']='True'

default_model = user_config['model']
model_choices = whisper.available_models() + ["Thai_Thonburian"]
label_model_size = "Model Size"
default_language = user_config['language']
auto = "Auto"
language_choices = [auto] + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()])
label_language = "Language"
transcribe_button_text = "Transcribe"
transcribed_output_text = "Transcription Output"


def check_cuda():
    print(f"GPU available: {torch.cuda.is_available()}")
    print(torch.cuda.get_device_name(0))


def process_custom_transformer(audio_path, language):
    MODEL_NAME = "biodatlab/whisper-th-medium-combined"
    device = 0 if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
    text = pipe(audio_path)["text"]
    return text


def file_to_text(tmp_file_name, model_size, language):
    check_cuda()
    language = None if (language == auto or len(language) == 0) else language
    save_config(model_size, language)

    if model_size == "Thai_Thonburian":
            return process_custom_transformer(tmp_file_name.name, language=language)
    
    model = whisper.load_model(model_size)
    result = model.transcribe(tmp_file_name.name, language=language)

    return result['text'].strip()


def youtube_to_text(url, model_size, language):
    check_cuda()
    language = None if (language == auto or len(language) == 0) else language
    save_config(model_size, language)
    
    audios = get_audio([url])
    for _, audio_path in audios.items():
        if model_size == "Thai_Thonburian":
            return process_custom_transformer(audio_path, language=language)
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, language=language)

    return result['text'].strip()


def save_config(model_size, language):
    user_config['model'] = model_size
    user_config['language'] = language
    with open('user_config.json', 'w') as f:
        json.dump(user_config, f)


with gr.Blocks() as demo:
    with gr.Tab("URL"):
        youtube_button = gr.Button(transcribe_button_text)
        youtube_button.click(youtube_to_text,
                                inputs=[
                                    gr.inputs.Textbox(label="YouTube URL"),
                                    gr.inputs.Dropdown(choices=model_choices, label=label_model_size, default=default_model),
                                    gr.inputs.Dropdown(choices=language_choices, label=label_language, default=default_language)
                            ], outputs=gr.Textbox(label=transcribed_output_text))

    with gr.Tab("File"):
        file_button = gr.Button(transcribe_button_text)
        file_button.click(file_to_text,
                            inputs=[
                                gr.inputs.File(label="Audio File"),
                                gr.inputs.Dropdown(choices=model_choices, label=label_language, default=default_model),
                                gr.inputs.Dropdown(choices=language_choices, label=label_language, default=default_language)
                            ],
                            outputs=gr.Textbox(label=transcribed_output_text))
demo.launch()
