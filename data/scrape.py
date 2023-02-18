import json
import requests

past_date = 1676730916

running = []
n = 0
while True:
    res = requests.get("https://api.pushshift.io/reddit/comment/search", dict(
        q="diagnosed depression", filter=",".join(["body", "author", "created_utc", "subreddit", "score", "id"]),
        limit=1000, until=past_date
    )).json()
    running.extend(res["data"])
    last = running[-1]["created_utc"]

    assert last <= past_date
    past_date = last

    if len(running) >= 50000:
        past_n = n
        n += len(running)
        with open(f'output/comments_{past_n}_{n}.jsonl', mode='w') as writer:
            for entry in running:
                json.dump(entry, writer)
                writer.write("\n")
        
        running.clear()
    
    if n > 10000000:
        break
