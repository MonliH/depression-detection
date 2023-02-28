import json
import pandas as pd
import requests
from tqdm import tqdm
import random


depressed_users = json.load(open("./output/depressed_comments_by_user.json", "r"))
non_depressed_users = json.load(open("./output/non_depressed_comments_by_user.json", "r"))

usernames_to_avoid = set(k.lower() for k in non_depressed_users.keys()).union(k.lower() for k in set(depressed_users.keys()))
all_usernames = pd.read_csv("output/users.csv")[::-1]

print(len(usernames_to_avoid), "usernames to avoid")

total = 0
running = []
searched = set()

try:
    for i, row in tqdm(all_usernames.iterrows(), total=all_usernames.shape[0]):
        user = row["author"]
        if user.lower() in usernames_to_avoid:
            continue

        get_oldest = bool(random.getrandbits(1))
        try:
            res = requests.get("https://api.pushshift.io/reddit/comment/search", dict(
                filter=",".join(["body", "created_utc", "subreddit", "score", "id"]), author=user,
                limit=250, order="asc" if get_oldest else "desc"
            ))
            j = res.json()
            recent_posts = j["data"]
            if get_oldest:
                recent_posts.reverse()
        except Exception as e:
            try:
                print(repr(res.text))
            except:
                print("ERROR:")
                print(e)
            continue

        running.append(dict(user=user, posts=recent_posts, depressed=False, oldest=get_oldest))
        if len(running) > 10000:
            prev = total
            total += len(running)

            with open(f'output/user_comments_control/user_comments_{prev}_{total}.jsonl', mode='w') as writer:
                for entry in running:
                    json.dump(entry, writer)
                    writer.write("\n")

            running.clear()

except KeyboardInterrupt:
    print("interruped, saving last batch")

if len(running) > 0:
    prev = total
    total += len(running)
    with open(f'output/user_comments_control/user_comments_{prev}_{total}.jsonl', mode='w') as writer:
        for entry in running:
            json.dump(entry, writer)
            writer.write("\n")
