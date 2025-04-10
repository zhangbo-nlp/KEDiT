#!/usr/bin/env python

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from utils.template import K2T_PROMPTS, T2K_PROMPTS, Template

from model.modeling_kdadapter import KDAdapterForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    bert_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained bert model or model identifier from huggingface.co/models"
        },
    )
    qformer_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained qformer model or model identifier from huggingface.co/models"
        },
    )
    language_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained large language model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    freeze_bert_model: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the bert model parameters or not."},
    )
    freeze_qformer_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the qformer model parameters or not."},
    )
    freeze_language_model: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the language model parameters or not."},
    )
    num_query_tokens: int = field(
        default=32, metadata={"help": "The number of query tokens."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the "
                "pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def main():
    # Parse input arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    template = Template

    # Load pretrained model, tokenizer, and image processor
    bert_tokenizer = AutoTokenizer.from_pretrained(model_args.bert_model_name_or_path)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        model_args.language_model_name_or_path
    )
    llm_tokenizer.pad_token = "<|end_of_text|>"

    model = KDAdapterForConditionalGeneration.from_bert_qformer_llm_pretrained(
        model_args.bert_model_name_or_path,
        model_args.qformer_name_or_path,
        model_args.language_model_name_or_path,
        num_query_tokens=model_args.num_query_tokens,
        cache_dir=model_args.cache_dir,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
    )
    model.language_model.config.use_cache = False  # not use for fine-tuning
    config = model.config

    max_bert_length = min(
        data_args.max_seq_length, config.bert_config.max_position_embeddings
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_bert_model:
        _freeze_params(model.bert_model)

    if model_args.freeze_qformer_model:
        _freeze_params(model.qformer_model)

    if model_args.freeze_language_model:
        _freeze_params(model.language_model)

    # Load and preprocess the dataset
    raw_datasets = load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, trust_remote_code=True
    )
    train_dataset = raw_datasets["train"]

    def collate_fn(examples):
        knowledge = list([example["knowledge"] for example in examples])
        k2t_prompts = []
        t2k_prompts = []
        input_part_targets_len = []
        for example in examples:
            system_text = template.system_format.format(content=template.system)
            human_text = template.user_format.format(content=random.choice(K2T_PROMPTS))
            assistant_text = template.assistant_format.format(
                content=example["knowledge"]
            )
            k2t_prompts.append(system_text + human_text + assistant_text)
            input_part_targets_len.append(
                len(llm_tokenizer.tokenize(system_text + human_text)) + 1
            )  # +1 is bos token

            system_text = template.system_format.format(
                content=f"{template.system} {example['knowledge']}"
            )
            human_text = template.user_format.format(content=random.choice(T2K_PROMPTS))
            t2k_prompts.append(system_text + human_text)

        bert_inputs = bert_tokenizer(
            knowledge,
            max_length=max_bert_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        llm_inputs = llm_tokenizer(
            k2t_prompts,
            max_length=data_args.max_seq_length + 100,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        t2k_llm_inputs = llm_tokenizer(
            t2k_prompts,
            max_length=data_args.max_seq_length + 100,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )

        # do not apply loss to the padding
        targets = llm_inputs.input_ids.masked_fill(
            llm_inputs.input_ids == llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        return {
            "bert_input_ids": bert_inputs.input_ids,
            "bert_attention_mask": bert_inputs.attention_mask,
            "llm_input_ids": llm_inputs.input_ids,
            "llm_attention_mask": llm_inputs.attention_mask,
            "t2k_llm_input_ids": t2k_llm_inputs.input_ids,
            "t2k_llm_attention_mask": t2k_llm_inputs.attention_mask,
            "labels": targets,
        }

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        bert_tokenizer.save_pretrained(
            os.path.join(training_args.output_dir, "bert_tokenizer")
        )
        llm_tokenizer.save_pretrained(
            os.path.join(training_args.output_dir, "llm_tokenizer")
        )
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
