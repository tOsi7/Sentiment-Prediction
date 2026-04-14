This project builds a sentiment classification system for Reddit text using feature engineering from multiple sentiment lexicons and compares a logistic regression baseline with a neural network model.

The goal was to evaluate how well traditional linear models perform against non-linear models when using handcrafted linguistic features

Features:
    Extracted sentiment signals from:
        Adjective lexicon
        Frequent word lexicon
        Subreddit-specific lexicon
    Engineered statistical features:
        Mean sentiment scores
        Positive/negative word counts
        Text length and structure features
    Implemented:
        Logistic Regression (baseline)
        Feedforward Neural Network (PyTorch)
    Used:
        Train / Validation / Test split
        Accuracy and Macro F1-score
        Confusion Matrix for error analysis

Models:

    Logistic Regression
        Single linear layer
        Fast and interpretable baseline
    Neural Network
        2 hidden layers (ReLU)
        Captures non-linear feature interactions