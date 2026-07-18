"""
Flask web application for Restaurant Rating Prediction.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.

NOTE: You must run `python src/train_model.py` at least once before
starting this app, so that models/rating_model.pkl and
models/encoders.pkl exist.
"""

import json
import os

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# --- Load model + encoders once at startup ---
MODEL_PATH = os.path.join(MODELS_DIR, "rating_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
    raise FileNotFoundError(
        "Model files not found. Run `python src/train_model.py` first "
        "to train the model before starting the app."
    )

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)


def encode_value(value, mapping):
    """Look up a target-encoded value, falling back to the global mean
    for categories the model has not seen before."""
    if value in mapping:
        return mapping[value]
    return mapping.get("__GLOBAL_MEAN__", 3.0)


def build_features(form):
    city = form.get("city", "").strip()
    cuisines = form.get("cuisines", "").strip()
    cost = float(form.get("cost", 0) or 0)
    price_range = int(form.get("price_range", 1) or 1)
    table_booking = 1 if form.get("table_booking") == "yes" else 0
    online_delivery = 1 if form.get("online_delivery") == "yes" else 0
    votes = float(form.get("votes", 0) or 0)
    cuisine_count = len(cuisines.split(",")) if cuisines else 1

    city_enc = encode_value(city, encoders["city_map"])
    cuisines_enc = encode_value(cuisines, encoders["cuisines_map"])

    features = [
        city_enc,
        cuisines_enc,
        cost,
        price_range,
        table_booking,
        online_delivery,
        votes,
        cuisine_count,
    ]
    return features


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        cities=encoders["top_cities"],
        cuisines=encoders["top_cuisines"],
        price_ranges=encoders["price_range_options"],
        cost_min=encoders["cost_min"],
        cost_max=encoders["cost_max"],
        metrics=metrics,
        algorithm=metrics.get("algorithm", "Random Forest Regression"),
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    features = build_features(request.form)
    prediction = model.predict([features])[0]
    prediction = round(float(prediction), 2)
    prediction = max(0.0, min(5.0, prediction))  # clip to valid rating range

    return render_template(
        "index.html",
        cities=encoders["top_cities"],
        cuisines=encoders["top_cuisines"],
        price_ranges=encoders["price_range_options"],
        cost_min=encoders["cost_min"],
        cost_max=encoders["cost_max"],
        metrics=metrics,
        algorithm=metrics.get("algorithm", "Random Forest Regression"),
        prediction=prediction,
        form_data=request.form,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API version, e.g.
    curl -X POST http://127.0.0.1:5000/api/predict \
      -H "Content-Type: application/json" \
      -d '{"city": "New Delhi", "cuisines": "North Indian", "cost": 800,
           "price_range": 2, "table_booking": "yes",
           "online_delivery": "no", "votes": 120}'
    """
    data = request.get_json(force=True)
    features = build_features(data)
    prediction = model.predict([features])[0]
    prediction = round(float(prediction), 2)
    prediction = max(0.0, min(5.0, prediction))
    return jsonify({"predicted_rating": prediction})


if __name__ == "__main__":
    app.run(debug=True)
