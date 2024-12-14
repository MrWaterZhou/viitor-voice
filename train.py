"""
Fine-tuning the library models for sequence to sequence.
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, DataCollatorForSeq2Seq
from transformers import \
    set_seed, \
    Seq2SeqTrainer, \
    HfArgumentParser, \
    Seq2SeqTrainingArguments, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint, PREFIX_CHECKPOINT_DIR

from viitor_voice.custom import Qwen2ForSnacLM, BartForSnacLM

logger = logging.getLogger(__name__)


class SaveTokenizer(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        encoder_path = os.path.join(checkpoint_folder, 'encoder_tokenizer')
        decoder_path = os.path.join(checkpoint_folder, 'decoder_tokenizer')

        model = kwargs["model"]
        encoder_config = AutoConfig.from_pretrained(model.encoder_path, trust_remote_code=True)
        encoder_tokenizer = AutoTokenizer.from_pretrained(model.encoder_path, trust_remote_code=True)
        encoder_config.save_pretrained(encoder_path)
        encoder_tokenizer.save_pretrained(encoder_path)

        decoder_config = AutoConfig.from_pretrained(model.decoder_path, trust_remote_code=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained(model.decoder_path, trust_remote_code=True)
        decoder_config.save_pretrained(decoder_path)
        decoder_tokenizer.save_pretrained(decoder_path)

        return control


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    encoder_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Please note: tokenizer must have bos and eos"}
    )
    decoder_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models. Please note: tokenizer must have bos and eos"}
    )
    model_path: str = field(default=None, metadata={"help": "Source language id for translation."})
    encoder_hidden_size: int = field(default=768, metadata={"help": "input dim for alignment layer"})
    decoder_hidden_size: int = field(default=768, metadata={"help": "output dim for alignment layer"})
    trainable_strategy: str = field(default=False,
                                    metadata={"help": "encoder+layernorm/encoder/align"})

    encoder_token_to_train: str = field(default=None, metadata={"help": "only finetune the embedding of these words"})
    decoder_token_to_train: str = field(default=None, metadata={"help": "xx"})
    do_align: bool = field(default=False, metadata={"help": "xx"})
    model_type: str = field(default=None, metadata={"help": "xx"})
    num_hidden_layers: int = field(default=None, metadata={"help": "xx"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_type: str = field(default='json', metadata={"help": "json, parquet, arrow"})
    data_files: List[str] = field(default=None, metadata={"help": "the regrex of files"}, )
    is_encoded: bool = field(default=False)
    source_col_name: str = field(default='text', metadata={"help": ".."})
    target_col_name: str = field(default='text', metadata={"help": ".."})
    streaming: bool = field(default=True)
    merge_inputs: bool = field(default=True)

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


def build_model(model_args: ModelArguments):
    if model_args.model_type == 'seq2seq':
        model = BartForSnacLM.from_pretrained(model_args.model_path)
        encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
        return encoder_tokenizer, decoder_tokenizer, model

    if model_args.model_type == 'decoder':
        if model_args.num_hidden_layers is not None:
            model = Qwen2ForSnacLM.from_pretrained(model_args.model_path,
                                                   torch_dtype=torch.bfloat16,
                                                   num_hidden_layers=model_args.num_hidden_layers)
        else:
            model = Qwen2ForSnacLM.from_pretrained(model_args.model_path,
                                                   torch_dtype=torch.bfloat16)
        encoder_tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, trust_remote_code=True)
        # for name, param in model.named_parameters():
        #    if ("lm_head" not in name) and ('embed_tokens' not in name):
        #        param.requires_grad = False
        return encoder_tokenizer, decoder_tokenizer, model


def build_dataset(data_args):
    raw_dataset = load_dataset(data_args.data_type, data_files=data_args.data_files, split='train',
                               num_proc=12)

    return raw_dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    encoder_tokenizer, decoder_tokenizer, model = build_model(model_args)

    # Get the language codes for input/target.
    with training_args.main_process_first():
        train_dataset = build_dataset(data_args)
        if not data_args.streaming:
            train_dataset = train_dataset.filter(
                lambda x: max(len(x['input_ids']), len(x['labels'])) < data_args.max_length, num_proc=24).shuffle()
        else:
            train_dataset = train_dataset.to_iterable_dataset(128)
            train_dataset = train_dataset.filter(
                lambda x: max(len(x['input_ids']), len(x['labels'])) < data_args.max_length).shuffle(buffer_size=10000)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=encoder_tokenizer,
        pad_to_multiple_of=8,
        return_tensors='pt')

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=None if model_args.model_type in {'seq2seq', 'decoder'} else [SaveTokenizer()]
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        if model_args.model_type in {'seq2seq', 'decoder'}:
            encoder_tokenizer.save_pretrained(training_args.output_dir)
        else:
            # save config and tokenizer
            output_dir = training_args.output_dir
            encoder_path = os.path.join(output_dir, 'encoder_tokenizer')
            decoder_path = os.path.join(output_dir, 'decoder_tokenizer')
            encoder_tokenizer.save_pretrained(encoder_path)
            decoder_tokenizer.save_pretrained(decoder_path)
            encoder_config = trainer.model.encoder.config
            encoder_config.save_pretrained(encoder_path)
            decoder_config = AutoConfig.from_pretrained(model.decoder_path)
            decoder_config.save_pretrained(decoder_path)

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
