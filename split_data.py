"""FIXED: split data/train.csv into a frozen 80% train / 20% holdout.

Run once. The outputs are committed so the holdout never changes.
Do not modify this file as part of the autoresearch loop.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path("data")
SRC = DATA / "train.csv"
TRAIN_OUT = DATA / "train_80pc.csv"
TEST_OUT = DATA / "test_20pc.csv"
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = parser.parse_args()

    if (TRAIN_OUT.exists() or TEST_OUT.exists()) and not args.force:
        sys.exit(f"Refusing to overwrite {TRAIN_OUT} / {TEST_OUT} (frozen split). Use --force.")

    df = pd.read_csv(SRC)
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["Survived"], random_state=SEED
    )
    train.sort_values("PassengerId").to_csv(TRAIN_OUT, index=False)
    test.sort_values("PassengerId").to_csv(TEST_OUT, index=False)

    for name, part in [("train_80pc", train), ("test_20pc", test)]:
        print(f"{name}: {len(part)} rows, survival rate {part['Survived'].mean():.3f}")


if __name__ == "__main__":
    main()
