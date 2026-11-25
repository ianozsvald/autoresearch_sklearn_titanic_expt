# Titanic sklearn baseline (for autoresearch)

A minimal scikit-learn model for the Titanic survival dataset. It is set up so an
automated research loop can improve the features and model while the scoring stays fixed.

Author note - directed by Ian, written by claude, created as a base repo for experiments in PLAYGROUP
using Karpathy's autoresearch idea. Can this code be automatically evolved (yes of course it can) to a better
score using sklearn, via claude, evaluated by the `evaluate.py` script?

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync          # creates .venv with Python 3.12, scikit-learn and pandas
```

## Workflow

```bash
uv run split_data.py   # once only: creates the frozen 80/20 split (already committed)
uv run train.py        # 5-fold CV diagnostic, then fit and save models/model.joblib
uv run evaluate.py     # prints "accuracy: 0.XXXX", writes results/score.json
```

`results/score.json` contains `{"metric": "accuracy", "value": ..., "n": 179}`.

## Data

| File | Rows | Purpose |
|------|------|---------|
| `data/train.csv` | 891 | Original labelled data, the source of both splits |
| `data/test.csv` | 418 | Unlabelled, **not used** |
| `data/train_80pc.csv` | 712 | Training split, the only data train.py may see |
| `data/test_20pc.csv` | 179 | Frozen holdout, the only data the metric is computed on |

`test.csv` has no `Survived` column, so it can't be scored. Instead, `split_data.py` makes a
stratified, seeded (`random_state=42`) 80/20 split of `train.csv`.
It won't overwrite an existing split unless you pass `--force`.

## The metric

**Accuracy on `data/test_20pc.csv`, higher is better.** `train.py` also prints 5-fold
stratified CV accuracy on the 80% split. That is a less noisy diagnostic and is not the target.

## Fixed vs evolvable

| File | Status | Role |
|------|--------|------|
| `split_data.py` | **FIXED** | Creates the frozen split |
| `evaluate.py` | **FIXED** | Loads `models/model.joblib`, scores the holdout, writes `results/score.json` |
| `features.py` | evolvable | `build_pipeline()` returns an unfitted sklearn Pipeline |
| `train.py` | evolvable | Fits the pipeline on `train_80pc.csv` and saves it with joblib |

Rules for the research loop:

1. Edit only `features.py` and `train.py`.
2. Never read `data/test_20pc.csv` from evolvable code.
3. The saved model must accept **raw** DataFrame rows (all columns except `Survived`) and
   return 0/1 from `.predict()`. Put all feature engineering inside the pipeline.
4. Define any custom transformers in `features.py` (an importable module, not `__main__`)
   so `joblib.load` can unpickle them.

## Baseline

Logistic regression on `Sex` and `Pclass` (one-hot encoded) and `Age` (median-imputed, scaled):

- CV accuracy: 0.7978 ± 0.0099
- Holdout accuracy: **0.7877**

Note that one holdout row is about 0.56 percentage points, so small changes in the score are
mostly noise.
