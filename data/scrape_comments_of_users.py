import json
import itertools
import requests
from tqdm import tqdm


depressed_users = json.load(open("./output/depressed_comments_by_user.json", "r"))
non_depressed_users = json.load(open("./output/non_depressed_comments_by_user.json", "r"))
print(
    len(depressed_users), "depressed users,",
    len(set(non_depressed_users.keys()).difference(set(depressed_users.keys()))), "non-depressed users"
)

total = 0
running = []
for user, depression_posts in tqdm(depressed_users.items()):
    earliest_date = min(depression_posts, key=lambda x: x[1])
    non_depressed_post_ids = set()
    if user in non_depressed_users:
        non_depressed_post_ids = set(v[0] for v in non_depressed_users[user])

    res = requests.get("https://api.pushshift.io/reddit/comment/search", dict(
        filter=",".join(["body", "created_utc", "subreddit", "score", "id"]), author=user,
        limit=min(250+len(non_depressed_post_ids), 1000), until=earliest_date
    ))
    try:
        j = res.json()
        recent_posts = j["data"]
    except:
        print(repr(res.text))
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
