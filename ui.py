import gradio as gr
import whisper
import torch


def speech_to_text(tmp_file_name, model_size):
    print(f"GPU available: {torch.cuda.is_available()}")
    print(torch.cuda.get_device_name(0))
    model = whisper.load_model(model_size)
    result = model.transcribe(tmp_file_name.name)

    return result['text']


gr.Interface(
    fn=speech_to_text,
    inputs=[
        gr.inputs.File(label="Audio File"),
        gr.inputs.Dropdown(choices=["tiny","base" ,"small", "medium", "large"], label="Model Size", default="base")
    ],
    outputs="text").launch()
