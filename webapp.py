"""
Fake News Detection Flask App
-----------------------------
This app loads a pre-trained Machine Learning model and TF-IDF vectorizer
to classify Indian news headlines as REAL or FAKE.
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
import joblib

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Base directory (project root)
APP_DIR = Path(__file__).resolve().parent

# Paths to model and vectorizer
MODEL_PATH = APP_DIR / 'models' / 'model.joblib'
VEC_PATH = APP_DIR / 'models' / 'vectorizer.joblib'

# Initialize Flask app
app = Flask(__name__)


# -------------------------------------------------------------------
# Load model and vectorizer
# -------------------------------------------------------------------

def load_artifacts():
    """Load trained model and vectorizer from disk."""
    if not MODEL_PATH.exists() or not VEC_PATH.exists():
        raise FileNotFoundError("Model or vectorizer not found. Run 'train.py' first.")
    clf = joblib.load(MODEL_PATH)
    vec = joblib.load(VEC_PATH)
    print("✅ Model and vectorizer loaded successfully.")
    return clf, vec


clf, vec = load_artifacts()


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route('/')
def index():
    """Render homepage with input form."""
    return render_template('index.html', title='Fake News Detector')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle headline form submission from the web interface.
    Accepts both form and JSON input.
    """
    # Try getting JSON first (API-like request)
    data = request.get_json(silent=True)
    headline = None

    if data and 'headline' in data:
        headline = data['headline']
    else:
        # Fallback to form field (HTML form)
        headline = request.form.get('headline', '')

    headline = (headline or '').strip()

    if not headline:
        # Return with error if empty
        return render_template('index.html', error='⚠️ Please enter a news headline.')

    # Vectorize input and make prediction
    X = vec.transform([headline])
    pred = clf.predict(X)[0]

    # NOTE: Check label mapping — update if reversed
    label = 'FAKE' if pred == 1 else 'REAL'

    return render_template(
        'result.html',
        title='Prediction Result',
        headline=headline,
        label=label
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for JSON-based predictions.
    Example request:
    {
        "headline": "Government launches new AI policy"
    }
    """
    data = request.get_json(silent=True)

    if not data or 'headline' not in data:
        return jsonify({'error': 'Provide JSON with key "headline"'}), 400

    headline = data['headline']
    X = vec.transform([headline])
    pred = clf.predict(X)[0]
    label = 'FAKE' if pred == 1 else 'REAL'

    return jsonify({'headline': headline, 'label': label})


# -------------------------------------------------------------------
# Run app
# -------------------------------------------------------------------

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False  # Change to True for development
    )
