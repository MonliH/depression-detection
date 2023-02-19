import datasets
from glob import glob
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, TrainingArguments, pipeline
from transformers.pipelines.pt_utils import KeyDataset

comments = datasets.load_dataset("json", data_files=glob("output/comments_*.jsonl"))

deberta_tok = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

cols_to_remove = comments["train"].column_names
cols_to_remove.remove("id")
cols_to_remove.remove("author")
cols_to_remove.remove("body")

comments["train"] = comments["train"].remove_columns(cols_to_remove)

logits = []
pipe = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", tokenizer=deberta_tok, device=0)
labels = ["about the speaker being diagnosed with depression", "other"]
for out in tqdm(pipe(KeyDataset(comments["train"], "body"), candidate_labels=labels, multi_label=False, batch_size=32), total=1000):
    logit = out["scores"]
    if out["labels"][0][0] == "o":
        logit = list(reversed(logit))

    logits.append(logit)

comments["train"] = comments["train"].add_column("logits", logits)
comments.save_to_disk("./output/processed_deberta")
