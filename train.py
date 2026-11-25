"""EVOLVABLE: train the pipeline on data/train_80pc.csv and save it.

Reports 5-fold stratified CV accuracy as a diagnostic, then fits on the full
80% split and saves models/model.joblib for evaluate.py.
Never read data/test_20pc.csv here.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from features import build_pipeline

TRAIN = Path("data/train_80pc.csv")
MODEL = Path("models/model.joblib")
SEED = 42


def main() -> None:
    df = pd.read_csv(TRAIN)
    X, y = df.drop(columns="Survived"), df["Survived"]

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"cv_accuracy: {scores.mean():.4f} +/- {scores.std():.4f} (5-fold on {len(df)} rows)")

    pipeline.fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL)
    print(f"saved {MODEL}")


if __name__ == "__main__":
    main()
