# <center>ViiTor-Voice</center>
### <center>An LLM based TTS Engine</center>

<p align="center">
  <img src="asserts/post_1.png" alt="Viitor-Voice Cover">
</p>

## Update
- **2024.12.14**:
  - [demo](https://huggingface.co/spaces/ZzWater/viitor-voice)
  - Adjusted model input by removing speaker embeddings (we found that existing open-source speaker models struggle to capture speaker characteristics effectively and have limited generalization capabilities).
  - Supports zero-shot voice cloning.
  - Supports both Chinese and English languages.
  
## Features

- **Lightweight Design**  

  The model is simple and efficient, compatible with most LLM inference engines. With only 0.5B parameters, it achieves extreme optimization of computational resources while maintaining high performance. This design allows the model to be deployed not only on servers but also on mobile devices and edge computing environments, meeting diverse deployment needs.

- **Real-time Streaming Output, Low Latency Experience**  

  The model supports real-time speech generation, suitable for applications that demand low latency. On the Tesla T4 platform, it achieves an industry-leading first-frame latency of 200ms, providing users with nearly imperceptible instant feedback, ideal for interactive applications requiring quick response.

- **Rich Voice Library**  

  Offers more than 300 different voice options, allowing you to choose the most suitable speech style according to your needs and preferences. Whether it’s a formal business presentation or casual entertainment content, the model provides perfect voice matching.

- **Flexible Speech Rate Adjustment**  

  The model supports natural variations in speech rate, allowing users to easily adjust it based on content requirements and audience preferences. Whether speeding up for efficient information delivery or slowing down to enhance emotional depth, it maintains natural speech fluency.

- **Zero-shot Voice Cloning (Under Research)**  

  Decoder-only architecture naturally supports Zero-shot cloning, with future support for rapid voice cloning based on minimal voice samples.

---

## Environment Setup

```commandline
git clone https://github.com/viitor-ai/viitor-voice.git
cd viitor-voice
conda create -n viitor_voice python=3.10
conda activate viitor_voice
pip install -r requirements.txt

### Due to the issue with vllm's tokenizer length calculation, the token limit cannot take effect.
python_package_path=`pip show pip | egrep Location | awk -F ' ' '{print $2}'`
cp viitor_voice/utils/patch.py $python_package_path/vllm/entrypoints/openai/logits_processors.py
```

---

## Inference
### Pretrained Models
- ~~[English](https://huggingface.co/ZzWater/viitor-voice-en)~~(deprecated)
- ~~[Chinese](https://huggingface.co/ZzWater/viitor-voice-chs)~~(deprecated)
- [Chinese & English](https://huggingface.co/ZzWater/viitor-voice-mix)
### Offline Inference

**For GPU users**
```python
from viitor_voice.inference.vllm_engine import VllmEngine
import torchaudio

tts_engine = VllmEngine(model_path="ZzWater/viitor-voice-mix")

## chinese example
ref_audio = "reference_samples/reference_samples/chinese_female.wav"
ref_text = "博士,您工作辛苦了!"
text_list = ["我觉得我还能抢救一下的!", "我…我才不要和你一起!"]
audios = tts_engine.batch_infer(text_list, ref_audio, ref_text)
for i, audio in enumerate(audios):
    torchaudio.save('test_chinese_{}.wav'.format(i), audios[0], 24000)


# english example
ref_audio = "reference_samples/reference_samples/english_female.wav"
ref_text = "At dinner, he informed me that he was a trouble shooter for a huge international organization."
text_list = ["Working overtime feels like running a marathon with no finish line in sight—just endless tasks and a growing sense that my life is being lived in the office instead of the real world."]
audios = tts_engine.batch_infer(text_list, ref_audio, ref_text)
for i, audio in enumerate(audios):
    torchaudio.save('test_english_{}.wav'.format(i), audios[0], 24000)    

```
**For CPU users**
```python
from viitor_voice.inference.transformers_engine import TransformersEngine
import torchaudio

tts_engine = TransformersEngine(model_path="ZzWater/viitor-voice-mix", device='cpu')

## chinese example
ref_audio = "reference_samples/reference_samples/chinese_female.wav"
ref_text = "博士,您工作辛苦了!"
text_list = ["我觉得我还能抢救一下的!", "我…我才不要和你一起!"]
audios = tts_engine.batch_infer(text_list, ref_audio, ref_text)
for i, audio in enumerate(audios):
    torchaudio.save('test_chinese_{}.wav'.format(i), audios[0], 24000)


# english example
ref_audio = "reference_samples/reference_samples/english_female.wav"
ref_text = "At dinner, he informed me that he was a trouble shooter for a huge international organization."
text_list = [" Working overtime feels like running a marathon with no finish line in sight", " Just endless tasks and a growing sense that my life is being lived in the office instead of the real world."]
audios = tts_engine.batch_infer(text_list, ref_audio, ref_text)
for i, audio in enumerate(audios):
    torchaudio.save('test_english_{}.wav'.format(i), audios[0], 24000)    

```
### Gradio Demo
```bash
python gradio_demo.py
```

### Demo Inference
- [ViiTor AI](https://www.viitor.io/text-to-speech)
### Streaming Inference (TODO)

---
## Training
- [example](./train_example.md)
## Join Our Community
[![Join Discord](https://img.shields.io/discord/your-discord-id?logo=discord&style=for-the-badge)](https://discord.gg/MbxgFn7BN8)

Have questions about the project? Want to discuss new features, report bugs, or just chat with other contributors? Join our Discord community!
## References

- [SNAC](https://github.com/hubertsiuzdak/snac)
- [mini-omni](https://github.com/gpt-omni/mini-omni)
- [open-gpt-4-o](https://laion.ai/notes/open-gpt-4-o/)
- [Qwen](https://huggingface.co/Qwen/Qwen2-0.5B)
- [cosyvoice](https://huggingface.co/FunAudioLLM/CosyVoice-300M)

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to share and modify the code of this project for non-commercial purposes, under the following conditions:

1. **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
2. **Non-Commercial**: You may not use the material for commercial purposes.

**Copyright Notice:**  
© 2024 Livedata. All Rights Reserved.

