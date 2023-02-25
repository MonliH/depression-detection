import json
import random
import requests
from tqdm import tqdm


depressed_users = json.load(open("./output/depressed_comments_by_user.json", "r"))
non_depressed_users = json.load(open("./output/non_depressed_comments_by_user.json", "r"))

non_depressed_usernames = set(non_depressed_users.keys()).difference(set(depressed_users.keys()))
non_depressed_usernames = list(non_depressed_usernames)
non_depressed_usernames.sort()
random.Random(42).shuffle(non_depressed_usernames)

print(
    len(depressed_users), "depressed users,",
    len(non_depressed_usernames), "non-depressed users"
)

total = 0
running = []
skip = 22482
for i, user in enumerate(tqdm(non_depressed_usernames)):
    if i <= skip:
        continue

    non_depressed_post_ids = set()
    non_depressed_post_ids = set(v[0] for v in non_depressed_users[user])

    try:
        res = requests.get("https://api.pushshift.io/reddit/comment/search", dict(
            filter=",".join(["body", "created_utc", "subreddit", "score", "id"]), author=user,
            limit=min(250+len(non_depressed_post_ids), 1000)
        ))
        j = res.json()
        recent_posts = j["data"]
    except Exception as e:
        try:
            print(repr(res.text))
        except:
            print("ERROR:")
            print(e)
        continue

    posts_without_depression_mention = [x for x in recent_posts if x["id"] not in non_depressed_post_ids]
    # limit to 250 entries
    posts_without_depression_mention = posts_without_depression_mention[:250]

    running.append(dict(user=user, posts=posts_without_depression_mention, depressed=True))
    if len(running) > 10000:
        prev = total
        total += len(running)

        with open(f'output/user_comments/user_comments_{prev}_{total}.jsonl', mode='w') as writer:
            for entry in running:
                json.dump(entry, writer)
                writer.write("\n")

        running.clear()

if len(running) > 0:
    prev = total
    total += len(running)
    with open(f'output/user_comments/user_comments_{prev}_{total}.jsonl', mode='w') as writer:
        for entry in running:
            json.dump(entry, writer)
            writer.write("\n")
