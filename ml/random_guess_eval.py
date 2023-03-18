import datasets
import evaluate
import numpy as np

ds = datasets.load_from_disk("/mnt/disks/persist/user_comments_text_filtered_2", keep_in_memory=True)
TO_BINARY = {
    0: 1, # depressed
    1: 0, # control 1 (not depressed)
    2: 0, # control 2 (random)
}

def to_binary(vs):
    vs["label"] = [TO_BINARY[l] for l in vs["depressed_label"]]
    return vs

val = ds["validation"].map(to_binary, batched=True, num_proc=32).remove_columns(["text", "depressed_label"])
p_positive = 0.739
guesses = np.random.choice([0,1], size=len(val), p=[1-p_positive, p_positive])
guesses_2 = np.ones([len(val)])
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")

print(f1.compute(references=val["label"], predictions=guesses, average="macro"))
print(accuracy.compute(references=val["label"], predictions=guesses))

print("majority guessing:")
print(f1.compute(references=val["label"], predictions=guesses_2, average="macro"))
print(accuracy.compute(references=val["label"], predictions=guesses_2))
