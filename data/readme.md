Order to run scripts:

1. `scrape_comments.py` - get initial comments searching for the string "diagnosed depression"
2. `classify.py` - classify comments using `rafalposwiata/deproberta-large-depression` (from Hugging Face)
3. `classify_zeroshot.py`- classify comments (zeroshot with DeBERTa) using `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (from Hugging Face)
4. `merge_classifications.py` - merge classifications with original comments
5. `classify_users.py` - split users who posted a comment matching the "diagnosed depression" string into (probably) `depressed` and (probably) `not_depressed`
6. `scrape_comments_of_users.py` - scrape most recent 250 comments of both `depressed` and `not_depressed` users (for a depressed user, scrape from the most recent comment that doesn't have anything to do with depression)
