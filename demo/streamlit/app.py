from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn import functional
import pandas as pd
import os
import praw
import dotenv
import streamlit as st

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
dotenv.load_dotenv(dotenv_path)
model_path = os.environ["MODEL_PATH"]

reddit = praw.Reddit()
reddit.read_only = True


class Comment(BaseModel):
    subreddit: str
    text: str


def get_comments(username):
    user_comments = reddit.redditor(username).comments.new(limit=250)
    comments = []
    for comment in user_comments:
        comments.append(
            Comment(subreddit=comment.subreddit.display_name, text=comment.body)
        )

    return comments


def process_comments(comments):
    return "\n\n".join(
        f"Post from /r/{comment.subreddit}:\n{comment.text}" for comment in comments
    )


device = os.environ.get("DEVICE", "cpu")


@st.cache_data
def get_model_and_tok():
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return (model, tokenizer)


model, tokenizer = get_model_and_tok()


def inference(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**tokens)
    return functional.softmax(logits.logits)


st.title("Early Depression Detection")
username = st.text_input("Reddit Username", placeholder="/u/monlih")

if username:
    comments = get_comments(username)
    data = pd.DataFrame(
        {
            "subreddit": [comment.subreddit for comment in comments],
            "text": [comment.text for comment in comments],
            "contains_depression": [
                "depression" in comment.text.lower() for comment in comments
            ],
        }
    )
    st.experimental_data_editor(data)
