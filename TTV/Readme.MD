This project builds a sentiment analysis model using both traditional machine learning and a PyTorch neural network. It compares feature-based methods with learned word embeddings.

---

## Models

**1. Logistic Regression (Baseline)**
- Lexicon-based + text features
- Scikit-learn model

**2. PyTorch Neural Network**
- Word embeddings (`nn.Embedding`)
- Simple feedforward neural network
- Trained with binary classification loss