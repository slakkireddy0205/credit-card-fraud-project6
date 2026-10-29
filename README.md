# Credit Card Fraud Detection

A machine learning API that predicts whether a credit card transaction is fraudulent or legitimate.

**Live API:** https://credit-card-fraud-project6.onrender.com/docs

(The free Render plan sleeps when idle, so the first request can take up to a minute.)

## Dataset

Kaggle "Credit Card Fraud Detection" (MLG-ULB): 284,807 transactions, of which only 492 (0.17%) are fraud. Features are `Time`, `Amount` and 28 anonymized columns (`V1` to `V28`).

## Approach

- EDA: no missing values and no text columns. Fraud is very rare, so accuracy is misleading and recall is the main metric.
- Stratified 80/20 train/test split. `Time` and `Amount` scaled with `StandardScaler`.
- Imbalance handled with `class_weight="balanced"`.
- Models: Logistic Regression and Random Forest. The Random Forest was tuned with GridSearchCV, scored on recall.

## Results (test set, 98 fraud cases)

| Model | Recall | Precision | Fake alarms | Missed frauds |
|---|---|---|---|---|
| Logistic Regression (balanced) | 0.92 | 0.06 | 1,386 | 8 |
| Random Forest (default, balanced) | 0.79 | 0.91 | 8 | 21 |
| Random Forest (tuned) | 0.89 | 0.26 | 249 | 11 |

The tuned Random Forest is the final model. It catches almost as much fraud as Logistic Regression with about 5 times fewer false alarms.

## API

- `GET /` returns a status message.
- `POST /predict` takes the 30 transaction fields and returns the prediction, a label and the fraud probability.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Run with Docker

```bash
docker build -t fraud-api .
docker run -d -p 8001:8000 fraud-api
```

## Tests and CI/CD

- `python -m pytest` runs 3 API tests.
- GitHub Actions runs the tests and builds the Docker image on every push to `main`.
- Render deploys the Docker image automatically.

## Tech stack

Python, pandas, scikit-learn, FastAPI, Pydantic, joblib, Docker, GitHub Actions, Render.