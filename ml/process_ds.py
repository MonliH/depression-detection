import datasets
datasets.disable_caching()
ds = datasets.load_from_disk("/mnt/disks/persist/user_comments", keep_in_memory=True)
ds = ds.filter(lambda x: len(x["posts"]) > 0, num_proc=64, keep_in_memory=True)

def format_post_as_text(posts):
    return "\n\n".join(f"Post from /r/{post['subreddit']}:\n{post['body']}" for post in posts)

def add_text_format(batch):
    batch["text"] = [format_post_as_text(sample) for sample in batch["posts"]]
    return batch

new_ds = ds.map(add_text_format, remove_columns=["user", "posts"], batched=True, keep_in_memory=True)

split = new_ds.train_test_split(test_size=0.15, seed=42, keep_in_memory=True)
test_validation = split["test"].train_test_split(test_size=0.5, seed=42, keep_in_memory=True)
ds_dict = datasets.DatasetDict({
    "train": split["train"],
    "validation": test_validation["train"],
    "test": test_validation["test"]
})
ds_dict.save_to_disk("/mnt/disks/persist/user_comments_text")
