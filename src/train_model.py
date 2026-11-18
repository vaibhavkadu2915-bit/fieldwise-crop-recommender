"""Train a multinomial crop classifier from soil_measures.csv."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "soil_measures.csv"
MODEL_PATH = ROOT / "models" / "crop_model.joblib"
FEATURES = ["N", "P", "K", "ph"]
TARGET = "crop"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add soil_measures.csv to data/."
        )

    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = data[FEATURES + [TARGET]].dropna()
    if data[TARGET].nunique() < 2:
        raise ValueError("Training data must contain at least two crop classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=42,
        stratify=data[TARGET],
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print(f"Rows used: {len(data)} | Crops: {data[TARGET].nunique()}")
    print(f"Accuracy: {accuracy_score(y_test, predicted):.4f}")
    print(f"Weighted F1: {f1_score(y_test, predicted, average='weighted'):.4f}")
    print(f"Macro F1: {f1_score(y_test, predicted, average='macro'):.4f}")
    print(classification_report(y_test, predicted, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
