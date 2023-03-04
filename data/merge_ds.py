import datasets
from datasets import concatenate_datasets
import os

storage_options={"project": "my-google-project", "token": os.environ["GCS_TOKEN"]}

depressed = datasets.load_dataset("json", data_files="output/user_comments_depressed/*.jsonl")
non_depressed = datasets.load_dataset("json", data_files="output/user_comments_non_depressed/*.jsonl")
control = datasets.load_dataset("json", data_files="output/user_comments_control/*.jsonl")

def process_depressed(batch):
    batch["depressed_label"] = [0]*len(batch["depressed"])
    return batch

def process_non_depressed(batch):
    batch["depressed_label"] = [1]*len(batch["depressed"])
    return batch

def process_control(batch):
    batch["depressed_label"] = [2]*len(batch["depressed"])
    return batch

depressed = depressed.map(process_depressed, batched=True, num_proc=16, batch_size=1000, remove_columns=["depressed"])
non_depressed = non_depressed.map(process_non_depressed, batched=True, num_proc=16, batch_size=1000, remove_columns=["depressed"]).cast(depressed["train"].features)
control = control.map(process_control, remove_columns=["oldest", "depressed"], batched=True, num_proc=16, batch_size=1000).cast(depressed["train"].features)

ds = concatenate_datasets([depressed["train"], non_depressed["train"], control["train"]])
ds.shuffle(seed=123)
ds.save_to_disk("./output/user_comments", storage_options=storage_options)
