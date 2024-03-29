import datasets
datasets.disable_caching()
ds = datasets.load_from_disk("/mnt/disks/persist/user_comments", keep_in_memory=True)
ds = ds.filter(lambda x: len(x["posts"]) > 0, num_proc=64, keep_in_memory=True)

# fmt: off
subreddits_to_remove = set(sr.lower() for sr in [
    "Anxiety", "anxietyhelp", "anxietysuccess", "anxietysupporters", "CPTSD", "dpdr",
    "HealthAnxiety", "OCD", "PanicAttack", "Phobia", "pureo", "ptsd", "socialanxiety",
    "OCD", "depression", "depressed", "depression_help", "depressionregiments", 
    "DepressionJournals", "DepressionRecovery", "dysthymia", "AnxietyDepression", 
    "adhd_anxiety", "ADHD", "AdultADHDSupportGroup", "ashhd", "SuicideWatch"
])
# fmt: on

def format_post_as_text(posts):
    return "\n\n".join(
        f"Post from /r/{post['subreddit']}:\n{post['body']}" 
        for post in posts if (
            post["subreddit"].lower() not in subreddits_to_remove and "depression" not in post["subreddit"] and " depression " not in post["body"].lower() and " depressed " not in post["body"].lower()
        )
    )

def add_text_format(batch):
    batch["text"] = [format_post_as_text(sample) for sample in batch["posts"]]
    return batch

new_ds = ds.map(add_text_format, remove_columns=["user", "posts"], batched=True, keep_in_memory=True)
new_ds = new_ds.filter(lambda x: len(x["text"]) > 0, num_proc=64, keep_in_memory=True)

split = new_ds.train_test_split(test_size=0.15, seed=42, keep_in_memory=True)
test_validation = split["test"].train_test_split(test_size=0.5, seed=42, keep_in_memory=True)
ds_dict = datasets.DatasetDict({
    "train": split["train"],
    "validation": test_validation["train"],
    "test": test_validation["test"]
})
ds_dict.save_to_disk("/mnt/disks/persist/user_comments_text_filtered_2")
