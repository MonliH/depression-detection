import datasets
import numpy as np

deberta_preds = datasets.load_from_disk("./output/processed_deberta")

def process_label(examples):
    examples["class"] = np.argmax(examples["logits"], axis=1).tolist()
    return examples

result = deberta_preds.map(process_label, batched=True, num_proc=8)
result.save_to_disk("./output/labeled_deberta")
