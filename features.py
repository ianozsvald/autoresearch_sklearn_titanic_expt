"""EVOLVABLE: feature extraction and model definition.

build_pipeline() must return an unfitted sklearn estimator that accepts the
raw Titanic DataFrame columns (everything except 'Survived') and predicts 0/1.
All feature engineering must live inside the pipeline, so the saved model
works directly on raw rows. Put any custom transformers in this module so
joblib can unpickle them.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["Sex", "Pclass"]
NUMERIC = ["Age"]


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("features", preprocess),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
