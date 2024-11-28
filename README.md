# <center>ViiTor-Voice</center>
### <center>An LLM based TTS Engine</center>

<p align="center">
  <img src="asserts/post.webp" alt="Viitor-Voice Cover">
</p>

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

## Output Samples

Below are examples of speech generated by this project:

- Example 1: [Female Voice - 1](asserts/female_normal.mp3)
- Example 2: [Male Voice - 1](asserts/male_normal.mp3)

---

## Environment Setup

```commandline
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
- [English](https://huggingface.co/ZzWater/viitor-voice-en)
### Offline Inference

```python
from viitor_voice.utils.offline_inference import OfflineInference
import torchaudio

tts_engine = OfflineInference(model_path='ZzWater/viitor-voice-en',
                              config_path='viitor_voice/inference_configs/en.json')
text_list = [
    "Isn't it fascinating to think about the endless possibilities that lie within the pages of a book. every time you turn a page, you're diving into a new world ripe with potential for discovery, and wonder what stories will you uncover today."]
# list valid speakers
print(tts_engine.prompt_map.keys())
audios = tts_engine.batch_infer(text_list=text_list, speaker=['1'], speed=2)
torchaudio.save('test.wav', audios[0], 24000)
```

### Demo Inference
- [ViiTor AI](https://www.viitor.io/text-to-speech)
### Streaming Inference (TODO)

---
## Training (TODO)
## Join Our Community
[![Join Discord](https://img.shields.io/discord/your-discord-id?logo=discord&style=for-the-badge)](https://discord.gg/MbxgFn7BN8)

Have questions about the project? Want to discuss new features, report bugs, or just chat with other contributors? Join our Discord community!
## References

- [SNAC](https://github.com/hubertsiuzdak/snac)
- [mini-omni](https://github.com/gpt-omni/mini-omni)
- [open-gpt-4-o](https://laion.ai/notes/open-gpt-4-o/)

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).  
You are free to share and modify the code of this project for non-commercial purposes, under the following conditions:

1. **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.
2. **Non-Commercial**: You may not use the material for commercial purposes.

**Copyright Notice:**  
© 2024 Livedata. All Rights Reserved.

