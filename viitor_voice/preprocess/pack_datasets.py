import sys
from dataclasses import dataclass, field
from typing import Optional, List

import numba
import numpy as np
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from transformers import \
    HfArgumentParser


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_type: str = field(default='json', metadata={"help": "json, parquet, arrow"})
    data_files: str = field(default=None, metadata={"help": "the regrex of files"})
    tokenizer_path: str = field(default=None, metadata={"help": "processed tokenizer path"})
    save_path: str = field(default='text', metadata={"help": ".."})
    eos_token_id: int = field(default=156008)

    preprocessing_num_workers: Optional[int] = field(
        default=24,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "A100: 4096"
                "4090: 2048"
            )
        },
    )


@numba.njit
def ffd_check(a: np.ndarray, c: int, n: int):
    # First-fit-decreasing bin packing
    # Check if a[] could fit in n bins with capacity c
    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing

    a = np.sort(a)[::-1]
    bins = np.full((n,), c, dtype=a.dtype)
    for size in a:
        not_found = True
        for idx in range(n):
            if bins[idx] >= size:
                bins[idx] -= size
                not_found = False
                break

        if not_found:
            return False

    return True


@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    # First-fit-decreasing bin packing (with result return)

    indices = np.argsort(a)[::-1]
    a = a[indices]

    bins = []
    bins_result = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break

        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])

    return bins_result


@numba.njit
def allocate(lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, similar to Multifit
    # https://en.wikipedia.org/wiki/Multifit_algorithm
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if ffd_check(lengths[start_index: start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        batch = ffd_with_result(lengths[start_index: start_index + l], c, start_index)
        assert len(batch) <= n
        if len(batch) < n:
            break

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch[rank])

    return result, s, len(result) * c * n


class MultipackDistributedBatchSampler(Sampler):
    """Unpadded length sampling using Multipack.
       Approximate (at most ~1.22x) the optimal solution of the identical-machines scheduling problem, which is NP-hard."""

    def __init__(
            self,
            batch_max_length: int,
            lengths: List[int],
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            seed: int = 0,
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.batch_max_length = batch_max_length
        self.lengths = lengths
        assert isinstance(self.lengths, np.ndarray)

        self.epoch = 0

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.default_rng(seed=self.seed + self.epoch).permutation(len(self.lengths))

        lengths = self.lengths[indices]
        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate(lengths=lengths,
                                                    lengths_cumsum=lengths_cumsum,
                                                    rank=self.rank,
                                                    c=self.batch_max_length,
                                                    n=self.num_replicas)

        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots

        return batches

    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_type: str = field(default='json', metadata={"help": "json, parquet, arrow"})
    data_files: str = field(default=None, metadata={"help": "the regrex of files"})
    save_path: str = field(default='text', metadata={"help": ".."})
    eos_token_id: int = field(default=156008)

    preprocessing_num_workers: Optional[int] = field(
        default=24,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


def build_dataset(data_args):
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_path)
    raw_dataset = load_dataset(data_args.data_type, data_files=data_args.data_files, split='train',
                               num_proc=10).shuffle()
    pad_id = -100

    def format_train(sample):
        input_text = '<|START_TEXT|>' + sample['text'] + '<|END_TEXT|>' + '<|START_AUDIO|>'
        output_text = ''.join(
            ['<|speech-{}|>'.format(i) if j % 7 != 0 else '<|SEP_AUDIO|><|speech-{}|>'.format(i) for j, i in
             enumerate(sample['target'])]) + '<|END_AUDIO|>'
        input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
        output_ids = tokenizer(output_text, add_special_tokens=False).input_ids
        input_ids = input_ids + output_ids
        labels = input_ids + output_ids
        labels[:120] = [-100] * len(labels[:120])
        return {"input_ids": input_ids, 'labels': labels}

    raw_dataset = raw_dataset.map(format_train, num_proc=24, remove_columns=list(raw_dataset.column_names))
    raw_dataset = raw_dataset.filter(lambda x: max(x['labels']) > 0, num_proc=24)

    def encode_fn(examples):
        source_ids = examples['input_ids']
        target_ids = examples['labels']
        source_length = [len(x) for x in source_ids]

        lengths = np.array(source_length)
        cumlengths = lengths.cumsum()
        batches, _, _ = allocate(lengths, cumlengths, 0, data_args.max_length, 1)

        input_ids = []
        labels = []
        position_ids = []
        for batch in batches:
            tmp_input_ids = []
            tmp_labels = []
            tmp_position_ids = []
            for idx in batch:
                x = source_ids[idx]
                y = target_ids[idx]
                tmp_input_ids.extend(x)
                tmp_labels.extend(y)
                tmp_position_ids.extend(list(range(len(x))))
            if len(tmp_input_ids) > data_args.max_length:
                tmp_input_ids = tmp_input_ids[:data_args.max_length]
                tmp_labels = tmp_labels[:data_args.max_length]
                tmp_position_ids = tmp_position_ids[:data_args.max_length]
            if len(tmp_input_ids) < data_args.max_length:
                pad_size = data_args.max_length - len(tmp_input_ids)
                tmp_input_ids = tmp_input_ids + [data_args.eos_token_id] * pad_size
                tmp_labels = tmp_labels + [pad_id] * pad_size
                tmp_position_ids = tmp_position_ids + [0] * pad_size

            input_ids.append(tmp_input_ids)
            labels.append(tmp_labels)
            position_ids.append(tmp_position_ids)
        return {'input_ids': input_ids, "labels": labels, "position_ids": position_ids}

    raw_dataset = raw_dataset.map(encode_fn,
                                  batched=True,
                                  batch_size=10000,
                                  num_proc=data_args.preprocessing_num_workers,
                                  remove_columns=list(raw_dataset.column_names))
    raw_dataset = raw_dataset.filter(lambda x: len(set(x['labels'])) > 5, num_proc=data_args.preprocessing_num_workers)
    return raw_dataset


if __name__ == '__main__':
    parser = HfArgumentParser([DataTrainingArguments])
    data_args, = parser.parse_args_into_dataclasses()
    dataset = build_dataset(data_args)
    save_path = data_args.save_path
    dataset.to_parquet(save_path)

