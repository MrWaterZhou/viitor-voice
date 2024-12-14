import os
import re
from io import BytesIO
from urllib.parse import urlparse
import requests
import torchaudio


def load_audio(source):
    def is_url(path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    if is_url(source):
        # 从 URL 加载音频
        response = requests.get(source)
        response.raise_for_status()  # 检查请求状态
        audio_data = BytesIO(response.content)  # 转为类文件对象
    else:
        # 从本地文件加载音频
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        audio_data = source  # 本地路径可以直接传递给 torchaudio.load

    # 使用 torchaudio 加载音频
    waveform, sample_rate = torchaudio.load(audio_data)
    return waveform, sample_rate


pattern = re.compile(r"<\|speech-(\d+)\|>")


def combine_sequences(first_elements, second_elements, third_elements):
    group_size = 7
    sequence = []

    second_index = 0
    third_index = 0

    for first in first_elements:
        group = [None] * group_size

        # Assign the first element
        group[0] = first

        # Assign the second and fifth elements if they exist
        if second_index < len(second_elements):
            group[1] = second_elements[second_index]
            second_index += 1
        if second_index < len(second_elements):
            group[4] = second_elements[second_index]
            second_index += 1

        # Assign the remaining elements from third_elements if they exist
        for j in [2, 3, 5, 6]:
            if third_index < len(third_elements):
                group[j] = third_elements[third_index]
                third_index += 1

        # Remove None values at the end of the group if the group is incomplete
        sequence.extend([x for x in group if x is not None])

    return sequence


def split_sequence(sequence):
    group_size = 7
    first_elements = []
    second_elements = []
    third_elements = []

    # Iterate over the sequence in chunks of 7
    for i in range(0, len(sequence), group_size):
        group = sequence[i:i + group_size]

        # Add elements to the respective lists based on their position in the group
        if len(group) >= 1:
            first_elements.append(group[0])
        if len(group) >= 5:
            second_elements.extend([group[1], group[4]])
        if len(group) >= 7:
            third_elements.extend([group[2], group[3], group[5], group[6]])
        else:
            third_elements.extend(group[2:])

    return first_elements, second_elements, third_elements
