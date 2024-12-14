import numpy as np
import torch
import torchaudio
from snac import SNAC
from vllm import LLM, SamplingParams
from viitor_voice.inference.common import combine_sequences, load_audio, pattern, split_sequence


class VllmEngine:
    def __init__(self, model_path):
        stop = ["<|END_AUDIO|>"]
        tensor_parallel_size = 1
        max_tokens = 4096
        self.sampling_params = SamplingParams(temperature=0.0, best_of=1, stop=stop,
                                              max_tokens=max_tokens, repetition_penalty=1.3, n=1,
                                              allowed_token_ids=list(range(151640, 156014)))
        self.model = LLM(model=model_path,
                         tokenizer=model_path,
                         trust_remote_code=True,
                         tensor_parallel_size=tensor_parallel_size,
                         dtype='bfloat16',
                         max_model_len=max_tokens,
                         # gpu_memory_utilization=self.vllm_config.get('gpu_memory_utilization', 0.5),
                         )
        self.snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to('cuda')

    def batch_infer(self, text_list, prompt_audio_path, prompt_text, flattened_snac_encode=None):
        if flattened_snac_encode is None:
            prompt_audio, sr = load_audio(prompt_audio_path)
            if sr != 24000:
                prompt_audio = torchaudio.functional.resample(prompt_audio, sr, 24000)

            snac_encode = self.snac_model.encode(prompt_audio[None,].to('cuda'))
            first_elements, second_elements, third_elements = \
                snac_encode[0].cpu().numpy().tolist(), snac_encode[1].cpu().numpy().tolist(), snac_encode[
                    2].cpu().numpy().tolist()
            flattened_snac_encode = combine_sequences(first_elements[0], second_elements[0], third_elements[0])
        prompt_snac_texts = ''.join(
            ['<|speech-{}|>'.format(i) if j % 7 != 0 else '<|SEP_AUDIO|><|speech-{}|>'.format(i) for
             j, i in
             enumerate(flattened_snac_encode)])

        prompts = [
            '<|START_TEXT|>' + prompt_text + x + '<|END_TEXT|>' + '<|START_AUDIO|>' + prompt_snac_texts + '<|SEP_AUDIO|>'
            for x in text_list]
        outputs = self.model.generate(prompts, self.sampling_params)
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            snac_tokens = pattern.findall(generated_text)
            snac_tokens = [int(x) for x in snac_tokens]
            results.append(snac_tokens)
        audios = self.batch_decode_audios(results)
        return audios

    def batch_decode_audios(self, snac_tokens_list):
        audios = []
        with torch.no_grad():
            for snac_tokens in snac_tokens_list:
                try:
                    first_elements, second_elements, third_elements = split_sequence(snac_tokens)
                    codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to('cuda') for x in
                             [first_elements, second_elements, third_elements]]
                    audio_hat_all = self.snac_model.decode(codes)[0].cpu()
                    audios.append(audio_hat_all.to(torch.float32))
                except:
                    audios.append('error')
                    print('error')
        return audios
