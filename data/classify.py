import datasets
from glob import glob

from transformers import Trainer, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

original = datasets.load_dataset("json", data_files=glob("output/comments_*.jsonl"))

deberta_tok = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
roberta_tok = AutoTokenizer.from_pretrained("rafalposwiata/deproberta-large-depression")

def tokenize(sample):
    sample["deberta_input_ids"] = deberta_tok(sample["body"], max_length=512, truncation=True, padding="max_length")["input_ids"]
    sample["roberta_input_ids"] = roberta_tok(sample["body"], max_length=512, truncation=True, padding="max_length")["input_ids"]
    return sample

comments = original.map(tokenize, num_proc=16, batched=True)

def modify_for_roberta(sample):
    sample["input_ids"] = sample["roberta_input_ids"]
    sample["labels"] = [0]*len(sample["input_ids"])
    return sample

cols_to_remove = comments["train"].column_names
cols_to_remove.remove("id")
cols_to_remove.remove("author")

roberta_input = comments.map(modify_for_roberta, remove_columns=cols_to_remove, batched=True, num_proc=16)
print(roberta_input)

# deberta_model = AutoModel.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
roberta_model = AutoModelForSequenceClassification.from_pretrained("rafalposwiata/deproberta-large-depression")
args = TrainingArguments(per_device_eval_batch_size=192, output_dir="/tmp/")
trained_roberta = Trainer(model=roberta_model, args=args)

logits = trained_roberta.predict(roberta_input["train"])[0]
column = logits.tolist()

roberta_input["train"] = roberta_input["train"].add_column("logits", column)
roberta_input.save_to_disk("./output/processed_2")
