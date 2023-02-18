import datasets
from glob import glob

from transformers import Trainer, AutoModel, AutoTokenizer

original = datasets.load_dataset("json", data_files=glob("output/comments_*.jsonl"))

deberta_tok = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
roberta_tok = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")

def tokenize(sample):
    sample["deberta_input_ids"] = deberta_tok(sample["body"], max_length=512, truncation=True, padding="max_length")["input_ids"]
    sample["roberta_input_ids"] = roberta_tok(sample["body"], max_length=512, truncation=True, padding="max_length")["input_ids"]
    return sample

comments = original.map(tokenize, num_proc=16, batched=True)

deberta_model = AutoModel.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
roberta_model = AutoModel.from_pretrained("rafalposwiata/deproberta-large-depression")
trained_roberta = Trainer(model=roberta_model)

print(trained_roberta.predict(comments["train"]["roberta_input_ids"][:100]))
