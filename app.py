from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

diabetes_model     = joblib.load(os.path.join(MODELS_DIR, 'diabetes_model.pkl'))
hypertension_model = joblib.load(os.path.join(MODELS_DIR, 'hypertension_model.pkl'))
heart_model        = joblib.load(os.path.join(MODELS_DIR, 'heart_disease_model.pkl'))

with open(os.path.join(MODELS_DIR, 'feature_info.json')) as f:
    feature_info = json.load(f)

FEATURES         = feature_info['features']
CATEGORICAL_COLS = set(feature_info['categorical_cols'])

# Fields removed from the form — supplied as neutral defaults so the model still receives them
FORM_DEFAULTS = {
    'income_level':      'Middle',
    'employment_status': 'Employed',
}

DISEASE_COLORS = {
    'diabetes':     '#B5541E',
    'hypertension': '#8C6820',
    'heart_disease': '#8C3A28',
}
DISEASE_LABELS = {
    'diabetes':     'Diabetes',
    'hypertension': 'Hypertension',
    'heart_disease': 'Heart Disease',
}


def get_proba(pipeline, df):
    """Return At-Risk probability, handling hard VotingClassifier which lacks predict_proba."""
    try:
        return float(pipeline.predict_proba(df)[0][1])
    except AttributeError:
        clf = pipeline.named_steps['classifier']
        X_t = pipeline.named_steps['preprocessor'].transform(df)
        probas = [
            est.predict_proba(X_t)[0][1]
            for est in clf.estimators_
            if hasattr(est, 'predict_proba')
        ]
        if probas:
            return float(np.mean(probas))
        pred = clf.predict(X_t)[0]
        return 0.75 if pred == 1 else 0.25


def risk_tier(pct):
    if pct <= 33:
        return 'Low',      '#4A7040'
    if pct <= 60:
        return 'Moderate', '#8C6820'
    return 'High', None   # caller fills disease colour


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        row = {}
        for feat in FEATURES:
            val = request.form.get(feat, '').strip()
            if not val:
                val = FORM_DEFAULTS.get(feat, '')
            if not val:
                return render_template('index.html',
                                       error=f'Missing value: {feat.replace("_", " ").title()}')
            row[feat] = [val] if feat in CATEGORICAL_COLS else [float(val)]

        df = pd.DataFrame(row)

        results = {}
        for key, model in [
            ('diabetes',     diabetes_model),
            ('hypertension', hypertension_model),
            ('heart_disease', heart_model),
        ]:
            prob = get_proba(model, df)
            pct  = round(prob * 100)
            tier, colour = risk_tier(pct)
            results[key] = {
                'label':         DISEASE_LABELS[key],
                'pct':           pct,
                'tier':          tier,
                'colour':        colour or DISEASE_COLORS[key],
                'disease_colour': DISEASE_COLORS[key],
            }

        return render_template('results.html', results=results)

    except Exception as exc:
        return render_template('index.html', error=str(exc))


if __name__ == '__main__':
    app.run(debug=True)
