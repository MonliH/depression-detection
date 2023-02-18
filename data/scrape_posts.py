import json
import requests
from tqdm import tqdm

past_date = 1676730916

running = []
n = 0

with tqdm() as pbar:
    while True:
        res = requests.get("https://api.pushshift.io/reddit/submission/search", dict(
            q="diagnosed depression", filter=",".join(["title", "selftext", "author", "created_utc", "subreddit", "score", "id"]),
            limit=1000, until=past_date
        )).json()
        running.extend(res["data"])
        last = running[-1]["created_utc"]

        assert last <= past_date
        past_date = last
        print(len(res["data"]), past_date, list(running[-1].keys()))

        if len(running) >= 50000 or len(res["data"]) == 0:
            pbar.update(len(running))
            past_n = n
            n += len(running)
            with open(f'output/posts_{past_n}_{n}.jsonl', mode='w') as writer:
                for entry in running:
                    json.dump(entry, writer)
                    writer.write("\n")
            
            running.clear()
        
        if n > 10000000 or len(res["data"]) == 0:
            break
