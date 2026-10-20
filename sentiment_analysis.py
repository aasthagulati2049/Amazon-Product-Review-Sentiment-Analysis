"""

AMAZON PRODUCT REVIEW SENTIMENT ANALYSIS

GOAL
----
Given the text of a product review, predict whether the customer was
POSITIVE (happy) or NEGATIVE (unhappy). 

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so charts save to file without needing a screen
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# A fixed random seed means you get the SAME result every run.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



def make_dataset(n_samples=5000):
    """
    Loads real Amazon product reviews from the Hugging Face Hub.
    Dataset: Amazon Polarity — 3.6 million reviews, labeled 0 (negative) or 1 (positive).
    """
    from datasets import load_dataset

    print("Downloading Amazon reviews dataset...")
    dataset = load_dataset("fancyzhx/amazon_polarity", split="train")

    # Take a balanced subset — equal positive and negative
    df = dataset.to_pandas()[["content", "label"]].rename(
        columns={"content": "review"}
    )

    # Sample equal amounts of each class
    pos = df[df.label == 1].sample(n_samples // 2, random_state=RANDOM_SEED)
    neg = df[df.label == 0].sample(n_samples // 2, random_state=RANDOM_SEED)

    return pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def build_pipeline():
    """
    A scikit-learn Pipeline chains steps so they run in order automatically.

    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",   # drop "the", "a", "is" ...
            ngram_range=(1, 2),     # unigrams + bigrams
            min_df=2,               # ignore words appearing in <2 reviews (noise)
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
    ])


def evaluate(model, X_test, y_test, outdir="."):
    """
    Accuracy alone can lie. We report a full picture and save two charts:
        - Confusion matrix: where the model is right vs wrong
        - ROC curve + AUC: how well it separates the two classes overall
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of "positive"

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}")
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred,
                                target_names=["negative", "positive"]))

    # ---- Chart 1: Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(f"{outdir}/results/confusion_matrix.png", dpi=120)
    plt.close(fig)

    # ---- Chart 2: ROC curve ----
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray")  # random-guess baseline
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{outdir}/results/roc_curve.png", dpi=120)
    plt.close(fig)

    return acc, roc_auc


def show_top_words(model, k=10):
    """
    BONUS interpretability: print the words the model thinks are the strongest
    signals of positive and negative sentiment. 
    """
    vec = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos = names[np.argsort(coefs)[-k:]][::-1]
    top_neg = names[np.argsort(coefs)[:k]]
    print(f"\nTop {k} POSITIVE signals: {', '.join(top_pos)}")
    print(f"Top {k} NEGATIVE signals: {', '.join(top_neg)}")

def main():
    print("=" * 60)
    print("Amazon Review Sentiment Analysis")
    print("=" * 60)

    # 1. Get data
    df = make_dataset()
    print(f"Dataset: {len(df)} reviews "
          f"({df.label.sum()} positive, {(df.label == 0).sum()} negative)")

    # 2. Split: train the model on 80%, test it on the unseen 20%.
    #    stratify keeps the pos/neg balance equal in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["label"],
        test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"],
    )

    # 3. Build + train
    model = build_pipeline()
    model.fit(X_train, y_train)

    # 4. Evaluate
    acc, roc_auc = evaluate(model, X_test, y_test, outdir=".")
    show_top_words(model)

    # 5. Try it on brand-new sentences to see it generalize
    print("\nLive predictions on new reviews:")
    samples = [
        "this product is amazing i love it",
        "broke after one day total waste of money",
        "for the price overall i am very happy would buy again",
    ]
    for s in samples:
        pred = model.predict([s])[0]
        prob = model.predict_proba([s])[0].max()
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"  [{label}  {prob:.2f}]  \"{s}\"")

    print("\nSaved charts: confusion_matrix.png, roc_curve.png")
    print("Done.")


if __name__ == "__main__":
    main()
