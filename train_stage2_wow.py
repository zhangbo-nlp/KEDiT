#!/usr/bin/env python
# coding=utf-8

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from utils.evaluator import DialogEvaluator
from utils.template import Template

from model.configuration_kdadapter import KDAdapterConfig
from model.modeling_kdadapter import KDAdapterForConditionalGeneration

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_path: str = field(
        metadata={"help": "Path to pretrained model"},
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
        default=True,
        metadata={"help": "Whether to freeze the qformer model parameters or not."},
    )
    freeze_language_model: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the language model parameters or not."},
    )
    num_query_tokens: int = field(
        default=16,
        metadata={
            "help": "The number of prompts in knowledge prompts and knowledge aware attention."
        },
    )
    attn_mode: bool = field(
        default=True, metadata={"help": "Whether to use the knowledge aware attention."}
    )
    attn_scalar: float = field(
        default=1.0, metadata={"help": "The scalar factor used in attention."}
    )
    ffn_mode: bool = field(
        default=True, metadata={"help": "Whether to use the knowledge aware adapter."}
    )
    ffn_bottleneck: int = field(
        default=512, metadata={"help": "The bottleneck size of the feed forward."}
    )
    ffn_adapter_scalar: float = field(
        default=1.0, metadata={"help": "The scalar factor used in feed forward."}
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
    bert_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_args.model_path, "bert_tokenizer")
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_args.model_path, "llm_tokenizer")
    )

    config = KDAdapterConfig.from_pretrained(model_args.model_path)
    config.llm_config.attn_mode = model_args.attn_mode
    config.llm_config.attn_query_len = model_args.num_query_tokens
    config.llm_config.attn_scalar = model_args.attn_scalar
    config.llm_config.ffn_mode = model_args.ffn_mode
    config.llm_config.ffn_bottleneck = model_args.ffn_bottleneck
    config.llm_config.ffn_adapter_scalar = model_args.ffn_adapter_scalar
    config.hidden_size = config.llm_config.hidden_size

    model = KDAdapterForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    max_bert_length = min(
        data_args.max_seq_length, config.bert_config.max_position_embeddings
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_bert_model:
        _freeze_params(model.bert_model)

    if model_args.freeze_qformer_model:
        _freeze_params(model.qformer)

    if model_args.freeze_language_model:
        _freeze_params(model.language_model)

    unfreeze_list = []
    if model_args.attn_mode:
        unfreeze_list.extend(["attn_prompt", "attn_know_gate"])

    if model_args.ffn_mode:
        unfreeze_list.extend(["ka_ffn_adapter"])

    if len(unfreeze_list) > 0:
        for name, param in model.language_model.named_parameters():
            if any(partial_name in name for partial_name in unfreeze_list):
                param.requires_grad = True

    # Load and preprocess the dataset
    raw_datasets = load_dataset(
        data_args.dataset_name, data_args.dataset_config_name, trust_remote_code=True
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = concatenate_datasets(
        [raw_datasets["valid_random"], raw_datasets["valid_topic"]]
    )

    def collate_fn(examples):
        knowledge = list([example["knowledge"] for example in examples])

        prompts = []
        input_part_targets_len = []
        for example in examples:
            text = template.system_format.format(content=template.system)
            for turn_idx, turn in enumerate(example["context"]):
                if turn_idx % 2 == 0:
                    text += template.user_format.format(content=turn)
                else:
                    text += template.assistant_format.format(content=turn)

            input_part_targets_len.append(len(llm_tokenizer.tokenize(text)) + 1)
            text += template.assistant_format.format(content=example["response"])
            prompts.append(text)

        bert_inputs = bert_tokenizer(
            knowledge,
            max_length=max_bert_length,
            padding="longest",
            return_tensors="pt",
            truncation=True,
        )
        llm_inputs = llm_tokenizer(
            prompts,
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
            "labels": targets,
        }

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
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

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:

        def predict_collate_fn(examples):
            knowledge = list([example["knowledge"] for example in examples])

            prompts = []
            for example in examples:
                text = template.system_format.format(content=template.system)
                for turn_idx, turn in enumerate(example["context"]):
                    if turn_idx % 2 == 0:
                        text += template.user_format.format(content=turn)
                    else:
                        text += template.assistant_format.format(content=turn)

                prompts.append(text)

            bert_inputs = bert_tokenizer(
                knowledge,
                max_length=max_bert_length,
                padding="longest",
                return_tensors="pt",
                truncation=True,
            )
            llm_inputs = llm_tokenizer(
                prompts,
                max_length=data_args.max_seq_length + 100,
                padding="longest",
                return_tensors="pt",
                truncation=True,
            )

            return {
                "bert_input_ids": bert_inputs.input_ids,
                "bert_attention_mask": bert_inputs.attention_mask,
                "llm_input_ids": llm_inputs.input_ids,
                "llm_attention_mask": llm_inputs.attention_mask,
            }

        del trainer
        del model
        torch.cuda.empty_cache()

        model = KDAdapterForConditionalGeneration.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        model.language_model.config.use_cache = True  # use for inference

        test_random_dataset = raw_datasets["test_random"]
        test_topic_dataset = raw_datasets["test_topic"]

        random_ref = test_random_dataset["response"]
        topic_ref = test_topic_dataset["response"]

        test_random_dataloader = DataLoader(
            test_random_dataset,
            collate_fn=predict_collate_fn,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        )
        test_topic_dataloader = DataLoader(
            test_topic_dataset,
            collate_fn=predict_collate_fn,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
        )

        training_args.eval_strategy = "no"
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
        )

        test_random_dataloader, test_topic_dataloader = trainer.accelerator.prepare(
            test_random_dataloader, test_topic_dataloader
        )

        def predict_hyp(eval_dataloader):
            hyp_list = []
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = trainer.accelerator.unwrap_model(model).generate(
                        llm_input_ids=batch["llm_input_ids"],
                        llm_attention_mask=batch["llm_attention_mask"],
                        bert_input_ids=batch["bert_input_ids"],
                        bert_attention_mask=batch["bert_attention_mask"],
                        pad_token_id=llm_tokenizer.pad_token_id,
                        eos_token_id=[
                            llm_tokenizer.eos_token_id,
                            llm_tokenizer.pad_token_id,
                        ],
                        max_new_tokens=128,
                        do_sample=True,
                        num_beams=4,
                        no_repeat_ngram_size=2,
                        renormalize_logits=True,
                        epsilon_cutoff=3e-4,
                        temperature=0.8,
                        top_p=1.0,
                        top_k=30,
                    )
                generated_tokens = trainer.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=llm_tokenizer.pad_token_id
                )
                generated_tokens = trainer.accelerator.gather_for_metrics(
                    generated_tokens
                )

                decoded_preds = llm_tokenizer.batch_decode(
                    generated_tokens.cpu().numpy(), skip_special_tokens=True
                )
                hyp_list.extend(decoded_preds)

            return hyp_list

        evaluator = DialogEvaluator(metric_name="bleu&rouge&f1&dist")
        random_hyp = predict_hyp(test_random_dataloader)
        random_results = evaluator.compute(random_hyp, random_ref, post_proc=True)

        if trainer.accelerator.is_main_process:
            print("Random:")
            print(random_results)
            with open(
                os.path.join(training_args.output_dir, "random_results.json"), "w"
            ) as f:
                json.dump(random_results, f)
            with open(
                os.path.join(training_args.output_dir, "random_hyp.txt"), "w"
            ) as f:
                f.write("\n".join(random_hyp))

        topic_hyp = predict_hyp(test_topic_dataloader)
        topic_results = evaluator.compute(topic_hyp, topic_ref, post_proc=True)
        if trainer.accelerator.is_main_process:
            print("Topic:")
            print(topic_results)
            with open(
                os.path.join(training_args.output_dir, "topic_results.json"), "w"
            ) as f:
                json.dump(topic_results, f)
            with open(
                os.path.join(training_args.output_dir, "topic_hyp.txt"), "w"
            ) as f:
                f.write("\n".join(topic_hyp))


if __name__ == "__main__":
    main()
