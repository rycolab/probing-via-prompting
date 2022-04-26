#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import random

import datasets
from datasets import load_dataset

import torch
import transformers
from transformers import (
    AdamW,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from modeling_gated_gpt2 import GatedGPT2LMHeadModel
from modeling_gpt2_pp import GPT2ForProbingViaPrompting
from trainer_pp import PPTrainer

from utils import LABEL_DICT, convert_gate_to_mask

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MAX_LENGTH = {'pos': 360, 'const': 700, 'ner': 360, 'coref': 340, 'srl': 685}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    gpt2_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    prefix_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to trained prefix model."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='cache/',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix_len: Optional[int] = field(
        default=200,
        metadata={"help": "Length of the prefix."},
    )
    prefix_dim: Optional[int] = field(
        default=512,
        metadata={"help": "Dimension of hidden states of prefix model."},
    )
    prefix_drop: Optional[float] = field(
        default=0.0,
        metadata={"help": "Droput rate for the prefix model."},
    )
    num_of_heads: Optional[int] = field(
        default=96,
        metadata={"help": "Number of heads left unpruned."},
    )
    pruning_lr: Optional[float] = field(
        default=0.1,
        metadata={"help": "Learning rate for head importance variables."},
    )
    do_prune: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether heads are pruned."},
    )
    head_mask_path: Optional[str] = field(
        default='None',
        metadata={
            "help": "Where head mask is stored"
        },
    )
    toggle_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether heads are pruned."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Where data is stored"}
    )
    task: Optional[str] = field(
        default='ner',
        metadata={"help": "Tasks, one or more of {pos, const, coref, ner, srl}."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}
    dataset_args = {}
    logger.info("Loading data for {}".format(data_args.task))
    if training_args.do_train:
        data_files["train"] = os.path.join(data_args.data_dir, data_args.task, 'train.json')
    data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
    if "_control" in data_args.task:
        data_args.task = data_args.task.replace("_control", "")

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.gpt2_name_or_path:
        config = AutoConfig.from_pretrained(model_args.gpt2_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.gpt2_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.gpt2_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    special_tokens_dict = {
        'sep_token': '<sep>',
        'additional_special_tokens': list(LABEL_DICT[data_args.task].values())
    }
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    config.num_labels = num_added_toks - 1
    if model_args.gpt2_name_or_path:
        gpt2 = GatedGPT2LMHeadModel.from_pretrained(
            model_args.gpt2_name_or_path,
            from_tf=bool(".ckpt" in model_args.gpt2_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        gpt2 = GatedGPT2LMHeadModel(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
        logger.info(f"Training new gpt2 from scratch - Total size={n_params/2**20:.2f}M params")

    gpt2.resize_token_embeddings(len(tokenizer))
    gpt2.eval_acc = True

    if model_args.prefix_model_path:
        config = AutoConfig.from_pretrained(model_args.prefix_model_path, cache_dir=model_args.cache_dir)
        model = GPT2ForProbingViaPrompting.from_pretrained(model_args.prefix_model_path, config=config, gpt2=gpt2)
    else:
        config.prefix_len = model_args.prefix_len
        config.prefix_dim = model_args.prefix_dim
        config.prefix_drop = model_args.prefix_drop
        model = GPT2ForProbingViaPrompting(config, gpt2)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        result = tokenizer(examples[text_column_name], padding="max_length", max_length=MAX_LENGTH[data_args.task])
        examples = {}
        examples['input_ids'] = result['input_ids']
        examples["labels"] = [
            [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(result["attention_mask"], result["input_ids"])]
        ]
        for i, elem in enumerate(examples['labels']):
            sep_idx = elem.index(tokenizer.eos_token_id) + 1
            examples['labels'][i][:sep_idx] = [-100] * sep_idx
        return examples

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if model_args.head_mask_path != 'None':
            head_mask = torch.load(model_args.head_mask_path)
            if model_args.toggle_mask:
                head_mask = (head_mask == 0).float()
            model.gpt2.apply_masks(head_mask)


    if training_args.do_train:
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": training_args.weight_decay,
                "lr": training_args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": training_args.learning_rate
            },
        ]

        if model_args.do_prune:
            model.gpt2.apply_dsp(model_args.num_of_heads)
            for n, p in model.named_parameters():
                if n == "gpt2.w":
                    p.requires_grad = True
            optimizer_grouped_parameters.append(
                {
                    "params": [p for n, p in model.named_parameters() if n == "gpt2.w"],
                    "lr": model_args.pruning_lr,
                }
            )
        
        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        optimizer = None

    # Initialize our Trainer
    trainer = PPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
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

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if model_args.do_prune:
        head_mask = convert_gate_to_mask(model.gpt2.w, model_args.num_of_heads)
        model.gpt2.apply_masks(head_mask)
        model.gpt2.use_dsp = False
        logger.info("Number of heads: {}".format(head_mask.sum()))
        logger.info(f'Number of heads in each layer: {head_mask.sum(-1)}')
        if training_args.output_dir is not None:
            torch.save(head_mask, os.path.join(training_args.output_dir, "mask" + str(model_args.num_of_heads) + ".pt"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        if model_args.head_mask_path != 'None':
            logger.info("Number of heads: {}".format(head_mask.sum()))

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
