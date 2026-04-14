import streamlit as st
import torch
import numpy as np

# load your trained model
# (you'll need to save it first — shown below)

@st.cache_resource
def load_model():
    model = torch.load("model.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# your feature extraction function (copy from your code)
def extract_features(text):
    # paste your fixed feature function here
    return np.array([...])  # make sure it returns 12 features

st.title("Reddit Sentiment Analyzer")

user_input = st.text_area("Enter text:")

if st.button("Predict"):
    features = extract_features(user_input)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(features)
        pred = torch.argmax(logits, dim=1).item()

    labels = ["Negative", "Neutral", "Positive"]
    st.write(f"Prediction: **{labels[pred]}**")