from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional
from typing import List
from dataclasses import dataclass
import os
import asyncpraw
import dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
dotenv.load_dotenv(dotenv_path)
model_path = os.environ["MODEL_PATH"]

reddit = asyncpraw.Reddit()
reddit.read_only = True

class Comment(BaseModel):
    subreddit: str
    text: str

async def get_comments(username):
    user_comments = (await reddit.redditor(username)).comments.new(limit=250)
    comments = []
    async for comment in user_comments:
        comments.append(Comment(subreddit=comment.subreddit.display_name, text=comment.body))

    return comments

def process_comments(comments):
    return "\n\n".join(f"Post from /r/{comment.subreddit}:\n{comment.text}" for comment in comments)

device = "cuda:0"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def inference(text):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**tokens)
    return functional.softmax(logits.logits)


app = FastAPI()

class User(BaseModel):
    username: str

@app.post("/get-posts")
async def get_posts(user_id: User) -> List[Comment]:
    username = user_id.username
    return await get_comments(username)

class Posts(BaseModel):
    posts: List[Comment]

@app.post("/predict")
async def predict(posts: Posts):
    text = process_comments(posts.posts)
    result = inference(text).tolist()[0]
    return result
