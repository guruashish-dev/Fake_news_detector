import argparse
from pathlib import Path

import joblib


def predict(headline: str, model_path: str, vec_path: str) -> str:
    clf = joblib.load(model_path)
    vec = joblib.load(vec_path)
    X = vec.transform([headline])
    pred = clf.predict(X)[0]
    return 'FAKE' if pred == 1 else 'REAL'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict whether a headline is FAKE or REAL')
    parser.add_argument('--headline', type=str, required=True)
    parser.add_argument('--model', type=str, default='models/model.joblib')
    parser.add_argument('--vectorizer', type=str, default='models/vectorizer.joblib')
    args = parser.parse_args()

    model_path = Path(args.model)
    vec_path = Path(args.vectorizer)
    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError('Model or vectorizer not found. Run train.py first.')

    label = predict(args.headline, model_path, vec_path)
    print(label)
