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
DATASET = Path(r"C:\Users\ISHVAJEET\Downloads\model3_cleaned_datasetCopy.csv")
OUT = ROOT / "models" / "model3_nfr_type_classifier.joblib"
LABELS = ["A", "PE", "SE", "US"]

EXTRA_ROWS = [
    ("The system must be available 24/7", "A"),
    ("Service uptime should be 99.9%", "A"),
    ("The system should always be accessible", "A"),
    ("Users must access the system anytime", "A"),
    ("Downtime must be minimal", "A"),
    ("The application should not go offline", "A"),
    ("Availability must be ensured at all times", "A"),
    ("The platform must remain accessible during peak usage", "A"),
    ("The interface should be intuitive", "US"),
    ("Users should easily navigate the system", "US"),
    ("The system should be user-friendly", "US"),
    ("Minimal training should be required", "US"),
    ("The UI should be simple and clear", "US"),
]


def main() -> None:
    texts, labels = read_csv_rows(DATASET)
    rows = [(text.strip(), label.strip().upper()) for text, label in zip(texts, labels)]
    rows = [(text, label) for text, label in rows if label in LABELS]
    rows.extend(EXTRA_ROWS)
    texts = [row[0] for row in rows]
    labels = [row[1] for row in rows]
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
                        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(classification_report(y_test, preds, labels=LABELS, target_names=LABELS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
