import sys

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from snac import SNAC
from tqdm import tqdm
from viitor_voice.inference.common import combine_sequences


def compute_mse(original, reconstructed):
    # Ensure both inputs are the same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]

    mse = torch.mean((original - reconstructed) ** 2)
    return mse


class DataPrepare:
    def __init__(self, snac_sampling_rate, snac_model_path):
        self.snac_sampling_rate = snac_sampling_rate
        self.snac_model = SNAC.from_pretrained(snac_model_path).eval().cuda()
        self.stripe = 4096

    def get_snac_encode(self, audio, sr):
        if sr != self.snac_sampling_rate:
            snac_audio = torchaudio.functional.resample(audio, sr, self.snac_sampling_rate)
        else:
            snac_audio = audio
        snac_audio = snac_audio.unsqueeze(0)
        audio_hat, snac_encode = self.snac_model(snac_audio)
        norm = compute_mse(audio_hat[0], snac_audio)
        first_elements, second_elements, third_elements = \
            snac_encode[0].cpu().numpy().tolist(), snac_encode[1].cpu().numpy().tolist(), snac_encode[
                2].cpu().numpy().tolist()
        flattened_snac_encode = combine_sequences(first_elements[0], second_elements[0], third_elements[0])
        return flattened_snac_encode, norm.cpu().numpy().tolist()

    @torch.no_grad()
    def __call__(self, audio, text, sr):
        snac_encode, norm, = self.get_snac_encode(audio, sr)
        res = {"text": text, 'target': snac_encode, 'norm': norm}
        return res


def emilia(file):
    from datasets import load_dataset
    import json
    handler = DataPrepare(24000, 'hubertsiuzdak/snac_24khz')
    ds = load_dataset('webdataset', data_files=file, split='train')
    ds = ds.filter(lambda x: x['json']['dnsmos'] > 3.3, num_proc=12)
    with open(file + '.json', 'w', encoding='utf8') as f:
        for sample in tqdm(ds):
            try:
                audio = torch.tensor(sample['mp3']['array'].astype('float32')).to('cuda')[None]
                sr = sample['mp3']['sampling_rate']
                text = sample['json']['text']
                data = handler(audio, text, sr)
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            except:
                print('error')


if __name__ == '__main__':
    emilia(sys.argv[1])
