import gradio as gr
import sys
from viitor_voice.inference.vllm_engine import VllmEngine

if __name__ == '__main__':
    # Initialize your OfflineInference class with the appropriate paths
    offline_inference = VllmEngine("ZzWater/viitor-voice-mix")


    def clone_batch(text_list, prompt_audio, prompt_text):
        print(prompt_audio.name)
        try:
            audios = offline_inference.batch_infer(
                text_list=[text_list],
                prompt_audio_path=prompt_audio.name,  # Use uploaded file's path
                prompt_text=prompt_text,
            )
            return 24000, audios[0].cpu().numpy()[0].astype('float32')
        except Exception as e:
            return str(e)


    with gr.Blocks() as demo:
        gr.Markdown("# TTS Inference Interface")
        with gr.Tab("Batch Clone"):
            gr.Markdown("### Batch Clone TTS")

            text_list_clone = gr.Textbox(label="Input Text List (Comma-Separated)",
                                         placeholder="Enter text1, text2, text3...")
            prompt_audio = gr.File(label="Upload Prompt Audio")
            prompt_text = gr.Textbox(label="Prompt Text", placeholder="Enter the prompt text")

            clone_button = gr.Button("Run Batch Clone")
            clone_output = gr.Audio(label="Generated Audios", type="numpy")

            clone_button.click(
                fn=clone_batch,
                inputs=[text_list_clone, prompt_audio, prompt_text],
                outputs=clone_output
            )

    demo.launch(server_name="0.0.0.0")
