# Spam Detection with TF-IDF and a Dense Neural Network

Builds a binary text classifier to identify spam emails. The pipeline combines TF-IDF vectorisation with a regularised Dense neural network trained in TensorFlow/Keras. Evaluation covers precision, recall, AUC, and a confusion matrix.

## Business Context

Spam filtering is a foundational NLP task. This project implements an end-to-end solution that goes from raw email text to a trained classifier, while handling the natural class imbalance typical in spam datasets.

## Dataset

`spam.csv` contains two columns: `Category` (ham / spam) and `Message` (raw email text).

## Methodology

**EDA:** Word cloud of the most frequent terms in spam messages, class distribution summary.

**Text Cleaning:** Removal of non-word characters, lowercase conversion, and whitespace normalisation via regex.

**Feature Extraction:** TF-IDF vectorisation with a vocabulary capped at 5,000 features. Labels are mapped to binary (ham=0, spam=1).

**Train/Test Split:** 80/20 stratified split with `random_state=1502`.

**Model Architecture:**
- Dense layer (50 units, ReLU activation)
- Dropout (rate=0.5) for regularisation
- Output Dense layer (1 unit, Sigmoid activation)

**Training:** Adam optimiser, binary cross-entropy loss, early stopping with patience=3 monitoring training loss.

**Metrics:** Accuracy, Precision, Recall, AUC. Classification report and confusion matrix on the held-out test set.

## Project Structure

```
03_spam_detection_tensorflow/
├── spam_detection.py    # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
tensorflow
wordcloud
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `spam.csv` in the same directory and run:

```bash
python spam_detection.py
```

Outputs: `spam_wordcloud.png`, `confusion_matrix.png`, `training_history.png`.
