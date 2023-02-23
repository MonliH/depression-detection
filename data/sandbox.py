import datasets
import numpy as np
import json


d = datasets.load_from_disk("./output/processed_deberta")
def process_logit(examples):
    examples["class"] = np.argmax(examples["logits"], axis=1).tolist()
    return examples

d = d.map(process_logit, batched=True, num_proc=8)
print(d["train"]["logits"])
neg = d.filter(lambda x: x["class"]==0)
pos = d.filter(lambda x: x["class"]==1)
print(neg, pos)

with open("neg.txt", "w") as f:
    for c in neg["train"]:
        f.write(c["body"])
        f.write("\n--------\n")

with open("pos.txt", "w") as f:
    for c in pos["train"]:
        f.write(c["body"])
        f.write("\n--------\n")
