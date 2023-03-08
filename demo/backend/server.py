from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional
from typing import List
from dataclasses import dataclass
import os
import asyncpraw
import dotenv

dotenv.load_dotenv(".env")
model_path = os.environ["MODEL_PATH"]

reddit = asyncpraw.Reddit()
reddit.read_only = True

class Comment(BaseModel):
    subreddit: str
    text: str

async def get_comments(username):
    try:
        user_comments = (await reddit.redditor(username)).comments.new(limit=250)
        comments = []
        async for comment in user_comments:
            comments.append(Comment(subreddit=comment.subreddit.display_name, text=comment.body))

        return comments
    except Exception as e:
        raise HTTPException(404, detail="Unable to fetch posts for user.")

def process_comments(comments):
    return "\n\n".join(f"Post from /r/{comment.subreddit}:\n{comment.text}" for comment in comments)

device = "cuda:0"
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def inference(text):
    tokens = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True).to(device)
    logits = model(**tokens)
    return functional.softmax(logits.logits)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
