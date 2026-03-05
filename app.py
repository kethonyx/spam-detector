import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import gradio as gr


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

data["message"] = data["message"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
X = vectorizer.fit_transform(data["message"])

y = data["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

pred = model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

msg = ["free money prize"]
msg_vec = vectorizer.transform(msg)

print("Prediction:", lr_model.predict(msg_vec))

print("Logistic Regression Results")
print(classification_report(y_test, lr_pred))


def predict_spam(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    prob = lr_model.predict_proba(vec)[0][1]
    pred = lr_model.predict(vec)[0]

    label = "Spam" if pred == 1 else "Not Spam"

    return f"{label} (confidence: {prob:.2f})"


interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=2, placeholder="Enter SMS message..."),
    outputs="text",
    title="SMS Spam Detection AI",
    description="Machine Learning model that detects spam messages using TF-IDF and Logistic Regression."
)

interface.launch()
