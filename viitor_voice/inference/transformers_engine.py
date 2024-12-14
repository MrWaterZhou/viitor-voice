import numpy as np
import torch
import torchaudio
from snac import SNAC
from transformers import AutoTokenizer, AutoModelForCausalLM
from viitor_voice.inference.common import combine_sequences, load_audio, pattern, split_sequence


class TransformersEngine:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.snac_model = SNAC.from_pretrained('hubertsiuzdak/snac_24khz').eval().to(device)

    def batch_infer(self, text_list, prompt_audio_path, prompt_text, flattened_snac_encode=None):
        if flattened_snac_encode is None:
            prompt_audio, sr = load_audio(prompt_audio_path)
            if sr != 24000:
                prompt_audio = torchaudio.functional.resample(prompt_audio, sr, 24000)

            snac_encode = self.snac_model.encode(prompt_audio[None,].to(self.device))
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
        prompt_ids_list = self.tokenizer(prompts, add_special_tokens=False).input_ids
        results = []
        for prompt_ids in prompt_ids_list:
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.int64).to(self.device)
            output_ids = self.model.generate(prompt_ids, eos_token_id=156008, no_repeat_ngram_size=0, num_beams=1,
                                             do_sample=False, repetition_penalty=1.3,
                                             suppress_tokens=list(range(151641)))
            output_ids = output_ids[0, prompt_ids.shape[-1]:].cpu().numpy().tolist()
            generated_text = self.tokenizer.batch_decode([output_ids], skip_special_tokens=False)[0]
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
                    codes = [torch.from_numpy(np.array(x).astype(np.int32)[None,]).to(self.device) for x in
                             [first_elements, second_elements, third_elements]]
                    audio_hat_all = self.snac_model.decode(codes)[0].cpu()
                    audios.append(audio_hat_all.to(torch.float32))
                except:
                    audios.append('error')
                    print('error')
        return audios
