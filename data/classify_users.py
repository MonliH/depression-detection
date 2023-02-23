import json
from tqdm import tqdm
import datasets
from glob import glob

depressed = {}
not_depressed = {}

ds = datasets.load_from_disk("./output/labeled_deberta")
comments = datasets.load_dataset("json", data_files=glob("output/comments_*.jsonl"))
ds["train"] = ds["train"].add_column("time", comments["train"]["created_utc"]).add_column("other_id", comments["train"]["id"])

for row in tqdm(ds["train"]):
    assert row["id"] == row["other_id"]

    a = row["author"]

    if row["class"]:
        if a not in not_depressed:
            not_depressed[a] = []
        not_depressed[a].append((row["id"], row["time"]))
    else:
        if a not in depressed:
            depressed[a] = []
        depressed[a].append((row["id"], row["time"]))

with open("output/depressed_comments_by_user.json", "w") as f:
    json.dump(depressed, f)

with open("output/non_depressed_comments_by_user.json", "w") as f:
    json.dump(not_depressed, f)
