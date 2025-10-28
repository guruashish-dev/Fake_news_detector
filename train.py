import argparse
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split


def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    return s.strip()


def main(data_path: str, output_dir: str, test_size: float, random_state: int):
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    # Expecting columns: 'headline' and 'label'
    if 'headline' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'headline' and 'label' columns")

    df['headline'] = df['headline'].astype(str).apply(clean_text)
    df = df[df['headline'].str.len() > 0]

    # Map labels to numeric: REAL -> 0, FAKE -> 1
    label_map = {'REAL': 0, 'FAKE': 1}
    df['label_num'] = df['label'].map(label_map)
    if df['label_num'].isna().any():
        raise ValueError('Found labels other than REAL/FAKE; please normalize labels')

    X = df['headline'].values
    y = df['label_num'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # TF-IDF vectorizer
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)

    # Logistic Regression classifier
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['REAL', 'FAKE'])
    cm = confusion_matrix(y_test, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    # Save artifacts
    model_path = output_dir / 'model.joblib'
    vec_path = output_dir / 'vectorizer.joblib'
    joblib.dump(clf, model_path)
    joblib.dump(vec, vec_path)

    print(f"Saved model to {model_path}")
    print(f"Saved vectorizer to {vec_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TF-IDF + LogisticRegression on headlines')
    parser.add_argument('--data', type=str, default='india_fake_news_dataset.csv', help='Path to CSV')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save model and vectorizer')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    main(args.data, args.output_dir, args.test_size, args.random_state)
