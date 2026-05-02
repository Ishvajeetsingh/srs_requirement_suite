from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.pipeline import read_csv_rows  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(r"C:\Users\ISHVAJEET\Downloads\clean_requirements_dataset.csv")
OUT = ROOT / "models" / "model2_fr_nfr_fallback.joblib"


def main() -> None:
    texts, raw_labels = read_csv_rows(DATASET)
    labels = [float(label) for label in raw_labels]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(classification_report(y_test, preds, target_names=["Functional", "Non-functional"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
