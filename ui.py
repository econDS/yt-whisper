import gradio as gr
import whisper
import torch
from yt_whisper.cli import get_audio

default_model = "base"
model_choices = ["tiny", "base", "small", "medium", "large"]
label_model_size = "Model Size"
transcribe_button_text = "Transcribe"
transcribed_output_text = "Transcription Output"


def check_cuda():
    print(f"GPU available: {torch.cuda.is_available()}")
    print(torch.cuda.get_device_name(0))


def file_to_text(tmp_file_name, model_size):
    check_cuda()
    model = whisper.load_model(model_size)
    result = model.transcribe(tmp_file_name.name)

    return result['text']


def youtube_to_text(url, model_size):
    check_cuda()
    model = whisper.load_model(model_size)
    audios = get_audio([url])
    for _, audio_path in audios.items():
        result = model.transcribe(audio_path)

    return result['text']


with gr.Blocks() as demo:
    with gr.Tab("File"):
        file_button = gr.Button(transcribe_button_text)
        file_button.click(file_to_text,
                            inputs=[
                                gr.inputs.File(label="Audio File"),
                                gr.inputs.Dropdown(choices=model_choices, label=label_model_size, default=default_model)
                            ],
                            outputs=gr.Textbox(label=transcribed_output_text))
        
    with gr.Tab("URL"):
        youtube_button = gr.Button(transcribe_button_text)
        youtube_button.click(youtube_to_text,
                                inputs=[
                                    gr.inputs.Textbox(label="Youtube URL"),
                                    gr.inputs.Dropdown(choices=model_choices, label=label_model_size, default=default_model)
                            ], outputs=gr.Textbox(label=transcribed_output_text))
demo.launch()