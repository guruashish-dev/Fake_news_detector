# India Fake News Detector (TF-IDF + LogisticRegression)

This small project trains a TF-IDF vectorizer and a Logistic Regression classifier to detect whether a headline is FAKE or REAL using `india_fake_news_dataset.csv`.

Files added:
- `train.py` - training script, saves `models/model.joblib` and `models/vectorizer.joblib`.
- `infer.py` - small inference script to predict a single headline.
- `requirements.txt` - minimal dependencies.

Usage

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Train the model (from project root where `india_fake_news_dataset.csv` is located):

```powershell
python train.py --data india_fake_news_dataset.csv --output_dir models
```

This prints test metrics and saves `models/model.joblib` and `models/vectorizer.joblib`.

3. Predict a headline:

```powershell
python infer.py --headline "Government announces new tax reforms for startups"
```

Notes
- Labels expected in the CSV are `REAL` and `FAKE` (case-sensitive). The script maps `REAL->0` and `FAKE->1`.
- This is a basic baseline. Consider improving performance with more preprocessing, balancing, and larger models.
