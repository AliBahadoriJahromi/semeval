# 🧠 SemEval Text Classification

This repository contains my solution for a **SemEval** task focused on **text classification**, using both **machine learning algorithms** and a **Convolutional Neural Network (CNN)**. The goal is to classify text data based on semantic or sentiment labels, following the format and evaluation metrics of a shared task from the **Semantic Evaluation (SemEval)** competition.

---

## 📚 About SemEval

**SemEval** (Semantic Evaluation) is a series of international NLP challenges that aim to evaluate semantic analysis systems across a variety of tasks such as:

- Sentiment analysis
- Emotion detection
- Semantic similarity
- Aspect-based classification
- Toxicity detection

This project is based on one of those shared tasks, where each sample of text (e.g., tweets, reviews, or comments) is labeled with a specific semantic or sentiment category.

---

## 🛠️ Approaches Used

Two major categories of models were implemented:

### 1. ⚙️ Classical Machine Learning Models
- **Logistic Regression**

These models use manually engineered features such as:
- TF-IDF vectorization
- Token/word-level statistics
- Custom preprocessing pipelines

### 2. 🧠 Convolutional Neural Network (CNN)
A deep learning model trained on tokenized and embedded text sequences. The CNN architecture includes:
- Embedding layer
- Convolution + ReLU
- MaxPooling
- Fully Connected layers with dropout

The CNN was trained using PyTorch and optimized using cross-entropy loss.

---