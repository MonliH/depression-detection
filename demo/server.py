import fastapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional
from dataclasses import dataclass
import sys
import praw

model_path = sys.argv[1]

reddit = praw.Reddit()
reddit.read_only = True

@dataclass
class Comment:
    subreddit: str
    text: str

def get_comments(username):
    user_comments = reddit.redditor(username).comments.new(limit=250)
    comments = []
    for comment in user_comments:
        comments.append(Comment(subreddit=comment.subreddit.display_name, text=comment.body))

    return comments

def process_comments(comments):
    return "\n\n".join(f"Post from /r/{comment.subreddit}:\n{comment.text}" for comment in comments)

text = process_comments(get_comments("a"))

device = "cpu"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def inference(text):
    tokens = tokenizer(text, return_tensors="pt")
    logits = model(**tokens)
    return functional.softmax(logits.logits)

