from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import numpy as np
import evaluate
import datasets

f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")
roc_auc = evaluate.load("roc_auc")

MAPPING = {
    0: 1, # depressed
    1: 0, # control 1 (not depressed)
    2: 0, # control 2 (random)
}
def map_depressed_label(ys):
    return [MAPPING[y] for y in ys]

samples = 30000
ds = datasets.load_from_disk("output/dataset-filtered-2/user_comments_text_filtered_2")
Xs = ds["train"]["text"][:samples]
Xs_val = ds["test"]["text"]
ys_val = map_depressed_label(ds["test"]["depressed_label"])

ys = map_depressed_label(ds["train"]["depressed_label"])[:samples]
print("done processing")

tf_idf_model = TfidfVectorizer(max_features=1000000)
tf_idf_model.fit(Xs)
Xs = tf_idf_model.transform(Xs)
print("done transforming")

n_estimators = 15
classifier = BaggingClassifier(SVC(probability=True, class_weight="balanced"), max_samples=1.0/n_estimators, n_estimators=n_estimators, bootstrap=False, n_jobs=-1)
classifier.fit(Xs, ys)

print("done fitting")

validation_features = tf_idf_model.transform(Xs_val)
print("done making features for validation set")

print("predicting on validation set")
y_preds_prob = classifier.predict_proba(validation_features)
y_pred = np.argmax(y_preds_prob, axis=1)
print(f1.compute(predictions=y_pred, references=ys_val, average="macro"))
print(accuracy.compute(predictions=y_pred, references=ys_val))

import json
json.dump({"preds": y_preds_prob.tolist(), "labels": ys_val}, open("tfidf_results.json", "w"))
