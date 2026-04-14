import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=12, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


@st.cache_resource
def load_model():
    model = LogisticRegression(input_dim=12, num_classes=3)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


def extract_features(text):
    words = tokenize(text)

    word_count = len(words)
    longest_word = max([len(w) for w in words]) if words else 1
    long_words = len([w for w in words if len(w) >= 5])


    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    f5 = 0
    f6 = 0
    f7 = 0
    f8 = 0
    f9 = 0
    f10 = np.log(word_count + 1)
    f11 = np.log(longest_word + 1)
    f12 = np.log(long_words + 1)

    return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12], dtype=np.float32)

st.title("Reddit Sentiment Analyzer")
st.write("Enter text to classify sentiment (Negative / Neutral / Positive)")

user_input = st.text_area("Input Text")

labels = ["Negative", "Neutral", "Positive"]


if st.button("Predict"):

    features = extract_features(user_input)
    features = torch.tensor(features).unsqueeze(0)

    with torch.no_grad():
        logits = model(features)
        pred = torch.argmax(logits, dim=1).item()

        probs = torch.softmax(logits, dim=1).numpy()[0]

    st.success(f"Prediction: {labels[pred]}")

    st.write("Confidence Scores:")
    st.write({
        "Negative": float(probs[0]),
        "Neutral": float(probs[1]),
        "Positive": float(probs[2])
    })


st.markdown("---")
st.caption("Built using PyTorch + Streamlit")