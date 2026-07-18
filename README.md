# Restaurant Rating Predictor — Web App

A full working web application that predicts a restaurant's **Aggregate Rating
(out of 5)**, trained on your uploaded restaurant dataset. Built with
**Flask** (backend + web server) and a plain **HTML/CSS** frontend — no
separate frontend framework needed.

The training script trains and compares **3 regression algorithms**:

| Algorithm | R² (test set) | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.514 | 0.388 | 0.303 |
| Decision Tree Regression | 0.606 | 0.349 | 0.259 |
| **Random Forest Regression** ✅ | **0.636** | **0.336** | **0.248** |

The best-performing algorithm (by R²) is automatically selected and used
by the web app for live predictions. All 3 results are saved to
`models/model_comparison.json` and plotted in `models/model_comparison.png`,
also shown on the web page itself.

---

## Project Folder Structure

```
restaurant-rating-app/
├── data/
│   └── dataset.xlsx              # Your dataset (already included)
│
├── src/
│   └── train_model.py            # Preprocessing + training + evaluation script
│
├── models/                       # Created automatically by train_model.py
│   ├── rating_model.pkl          # Best-performing trained model (auto-selected)
│   ├── encoders.pkl              # Category encoders + dropdown option lists
│   ├── metrics.json              # MSE, RMSE, MAE, R² for the best model
│   ├── model_comparison.json     # MSE, RMSE, MAE, R² for ALL 3 algorithms
│   ├── model_comparison.png      # Bar chart comparing R² across algorithms
│   └── feature_importance.png    # Feature importance chart (best model)
│
├── templates/
│   └── index.html                # Web page (form + result display)
│
├── static/
│   ├── style.css                 # Page styling
│   └── feature_importance.png    # Copied here at train time for display
│
├── app.py                        # Flask application (routes + prediction logic)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## How to Run in VS Code

### Step 1 — Open the folder
Open the `restaurant-rating-app` folder in VS Code (`File > Open Folder`).

### Step 2 — Create and activate a virtual environment
Open the VS Code terminal (`` Ctrl+` ``):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Train the model
This reads `data/dataset.xlsx`, cleans it, engineers features, trains the
model, evaluates it, and saves everything needed by the web app into `models/`.

```bash
python src/train_model.py
```

You should see output like:
```
Evaluation metrics: {'mse': 0.11, 'rmse': 0.34, 'mae': 0.25, 'r2': 0.64}
Done. Model + encoders saved in /models
```

### Step 5 — Copy the chart into static (only needed once, already done)

```bash
cp models/feature_importance.png static/feature_importance.png
```

### Step 6 — Run the web app

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

You'll see a form where you can select a city, cuisine, cost, price range,
table booking / online delivery, and votes — submit it to get a predicted
rating, shown with a star display, plus the feature-importance chart.

---

## How It Works

1. **Preprocessing (`src/train_model.py`)**
   - Drops rows with missing target rating and "Not rated" restaurants (rating = 0 with no real reviews).
   - Fills missing `Cuisines` values with `"Unknown"`.
   - Converts `Has Table booking` / `Has Online delivery` to binary (1/0).
   - Creates a `Cuisine_Count` feature (number of cuisines offered).
   - **Target-encodes** `City` and `Cuisines` (high-cardinality categorical
     columns) by mapping each category to the average rating seen in
     training data — unseen values at prediction time fall back to the
     global average rating, so the app never breaks on new input.

2. **Algorithms trained and compared** (all on the identical train/test split):
   - `LinearRegression` (scikit-learn)
   - `DecisionTreeRegressor` (max_depth=8, to limit overfitting)
   - `RandomForestRegressor` (300 trees)

   The one with the highest R² on the test set is automatically chosen
   and saved as `models/rating_model.pkl` — the one the web app actually uses.

3. **Evaluation metrics**: MSE, RMSE, MAE, R² computed for all 3 algorithms
   — saved to `models/model_comparison.json`, with the winning model's
   metrics also saved separately to `models/metrics.json` and shown at
   the top of the web page.

4. **Feature importance**: extracted from the winning model (tree-based
   models use `feature_importances_`; Linear Regression would use absolute
   coefficient values) — displayed on the web page so you can see which
   factors (e.g. Votes, Cost, City) influence rating predictions most.

5. **Web app (`app.py`)**: Loads the saved model once at startup. The `/`
   route renders the form; `/predict` handles form submissions and displays
   the result on the same page; `/api/predict` exposes a JSON API for
   programmatic use (e.g., `curl`, Postman, or another frontend).

---

## Retraining

If you update `data/dataset.xlsx`, just re-run:

```bash
python src/train_model.py
cp models/feature_importance.png static/feature_importance.png
```

then restart `python app.py`.

## Notes

- This uses Flask's built-in development server, which is fine for local use
  and demos. For deployment, use a production WSGI server (e.g. `gunicorn`).
- `Votes` is typically the single most influential feature — a restaurant's
  review count is a strong signal of its rating in this dataset. Keep that
  in mind when interpreting predictions for brand-new restaurants with 0 votes.
