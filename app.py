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


class NeuralNetModel(nn.Module):
    def __init__(self, input_dim=12, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)



@st.cache_resource
def load_models():
    model = LogisticRegression()
    model.load_state_dict(torch.load("logreg_model.pt", map_location="cpu"))
    model.eval()

    nn_model = NeuralNetModel()
    nn_model.load_state_dict(torch.load("nn_model.pt", map_location="cpu"))
    nn_model.eval()

    return lr_model, nn_model

lr_model, nn_model = load_models()


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def extract_features(text):
    words = tokenize(text)

    word_count = len(words)
    longest_word = max([len(w) for w in words]) if words else 1
    long_words = len([w for w in words if len(w) >= 5])

    return np.array([
        0,0,0,0,0,0,0,0,0,
        np.log(word_count + 1),
        np.log(longest_word + 1),
        np.log(long_words + 1)
    ], dtype=np.float32)


st.title("Reddit Sentiment Analyzer")

model_choice = st.radio("Choose Model", ["Logistic Regression", "Neural Network"])

text = st.text_area("Enter text")

labels = ["Negative", "Neutral", "Positive"]


if st.button("Predict"):

    features = torch.tensor(extract_features(text)).unsqueeze(0)

    model = lr_model if model_choice == "Logistic Regression" else nn_model

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
st.caption("Built with PyTorch + Streamlit")