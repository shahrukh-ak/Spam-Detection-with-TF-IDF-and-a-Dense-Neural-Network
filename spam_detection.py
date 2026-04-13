"""
Spam Detection with TF-IDF and a Dense Neural Network
=======================================================
Trains a binary text classifier to distinguish spam from ham (non-spam)
emails. The pipeline covers EDA with word clouds, TF-IDF feature
extraction, a regularised Dense neural network, and evaluation via
a classification report and confusion matrix.

Dataset: spam.csv  (columns: Category, Message)
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ── Data Loading and EDA ──────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the spam dataset and print class distribution."""
    df = pd.read_csv(filepath)
    print("Class distribution:")
    print(df["Category"].value_counts())
    return df


def plot_wordcloud(df: pd.DataFrame):
    """Generate a word cloud of the most common terms in spam messages."""
    spam_text = " ".join(df[df["Category"] == "spam"]["Message"].tolist())
    wc = WordCloud(width=800, height=400, background_color="black").generate(spam_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Most Common Words in Spam Messages", color="white", fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig("spam_wordcloud.png", dpi=150, facecolor="black")
    plt.show()
    print("Saved: spam_wordcloud.png")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove non-word characters, lowercase, and collapse whitespace."""
    text = re.sub(r"\W", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame, max_features: int = 5000, test_size: float = 0.2,
               random_state: int = 1502):
    """
    Clean messages, encode labels, apply TF-IDF vectorisation,
    and split into train/test sets.
    """
    df = df.copy()
    df["Message"] = df["Message"].apply(clean_text)
    df["Label"] = df["Category"].map({"ham": 0, "spam": 1})

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["Message"]).toarray()
    y = df["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
    return X_train, X_test, y_train, y_test, vectorizer


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> Sequential:
    """
    Dense neural network with a single hidden layer and dropout regularisation
    for binary spam classification.
    """
    model = Sequential([
        Dense(50, activation="relu", input_dim=input_dim),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    model.summary()
    return model


def train_model(model, X_train, y_train, epochs: int = 10,
                batch_size: int = 64, patience: int = 3):
    """Train with early stopping on training loss."""
    early_stop = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    return history


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """Print classification report and display confusion matrix."""
    y_prob = model.predict(X_test)
    y_pred = (y_prob > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot(colorbar=False)
    plt.title("Spam Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def plot_training_history(history):
    """Plot loss, precision, recall, and AUC across training epochs."""
    metrics = ["loss", "precision", "recall", "auc"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        ax.plot(history.history[metric], label="Train")
        if f"val_{metric}" in history.history:
            ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.suptitle("Training History", fontsize=14)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("Saved: training_history.png")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "spam.csv"

    df = load_data(DATA_PATH)
    plot_wordcloud(df)

    X_train, X_test, y_train, y_test, vectorizer = preprocess(df)

    model = build_model(input_dim=X_train.shape[1])
    history = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test)
    plot_training_history(history)
