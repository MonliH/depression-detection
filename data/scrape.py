import jsonlines
from pmaw import PushshiftAPI

api = PushshiftAPI()
gen = api.search_comments(q="diagnosed depression", limit=1000000)

with jsonlines.open('output/comments.jsonl', mode='w') as writer:
    for comment in gen:
        value = {}
        value["text"] = comment["body"]
        value["author"] = comment["author"]
        value["created_utc"] = comment["created_utc"]
        value["subreddit"] = comment["subreddit"]
        value["score"] = comment["score"]
        value["id"] = comment["id"]
        writer.write(value)
