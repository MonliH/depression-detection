import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import datasets
import evaluate
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import struct, traverse_util
from flax.jax_utils import pad_shard_unpad, replicate, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSequenceClassification,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    is_wandb_available,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry

from train import ModelArguments, DataTrainingArguments, glue_eval_data_collator, create_train_state, create_learning_rate_fn

@dataclass
class DataTrainingArgumentsEval(DataTrainingArguments):
    eval_set: str = field(
        default="test"
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArgumentsEval, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    num_labels = 2
    is_regression = False

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=not model_args.use_slow_tokenizer,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = FlaxAutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        dtype=jnp.dtype("bfloat16") if training_args.bf16 else jnp.dtype("float32"),
        config=config,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    TO_BINARY = {
        0: 1, # depressed
        1: 0, # control 1 (not depressed)
        2: 0, # control 2 (random)
    }

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["text"][:50000], padding="max_length", max_length=data_args.max_seq_length, truncation=True)

        result["labels"] = [TO_BINARY[sample] for sample in examples["depressed_label"]]
        return result


    raw_datasets = datasets.load_from_disk(data_args.load_dataset_from_disk)
    eval_dataset = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets.column_names
    )
    
    def eval_step(state, batch):
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        return logits

    p_eval_step = jax.pmap(eval_step, axis_name="batch")

    learning_rate_fn = create_learning_rate_fn(1,1,1,1,1)
    state = create_train_state(
        model, learning_rate_fn, is_regression, num_labels=num_labels, weight_decay=training_args.weight_decay
    )

    metric = evaluate.load("f1")
    metric2 = evaluate.load("accuracy")
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()

    print(eval_dataset)
    state = replicate(state)
    eval_loader = glue_eval_data_collator(eval_dataset, eval_batch_size)
    all_predictions = []
    all_labels = []
    for batch in tqdm(
        eval_loader,
        total=math.ceil(len(eval_dataset) / eval_batch_size),
        desc="Evaluating ...",
        position=2,
    ):
        labels = batch.pop("labels")
        predictions = pad_shard_unpad(p_eval_step)(
            state, batch, min_device_batch=per_device_eval_batch_size
        )
        predictions_np = np.array(predictions)
        # metric.add_batch(predictions=predictions_np, references=labels)
        # metric2.add_batch(predictions=predictions_np, references=labels)
        all_predictions.append(predictions_np)
        all_labels.append(labels)

    # eval_metric = {**metric.compute(average="macro"), **metric2.compute()}
    p = []
    l = []
    for pred, label in zip(all_predictions, all_labels):
        p.extend(pred.tolist())
        l.extend(label.tolist())

    import json
    json.dump({"pred": p, "label": l}, open("preds_text.json", "w"))

if __name__ == "__main__":
    main()
