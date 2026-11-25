"""FIXED: score the saved model on the frozen holdout.

Loads models/model.joblib and reports accuracy on data/test_20pc.csv. This
single number is what the autoresearch loop optimises.
Do not modify this file as part of the autoresearch loop.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL = Path("models/model.joblib")
HOLDOUT = Path("data/test_20pc.csv")
RESULTS = Path("results")


def main() -> None:
    model = joblib.load(MODEL)

    holdout = pd.read_csv(HOLDOUT)
    X, y = holdout.drop(columns="Survived"), holdout["Survived"]
    accuracy = float(accuracy_score(y, model.predict(X)))

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "score.json").write_text(
        json.dumps({"metric": "accuracy", "value": accuracy, "n": len(holdout)}, indent=2) + "\n"
    )

    print(f"accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
