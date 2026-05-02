from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.pipeline import read_csv_rows  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(r"C:\Users\ISHVAJEET\Downloads\requirement_detection_final.csv")
OUT = ROOT / "models" / "model1_requirement_detector.joblib"


def main() -> None:
    texts, raw_labels = read_csv_rows(DATASET)
    labels = [int(float(label)) for label in raw_labels]
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced", dual="auto"),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(classification_report(y_test, preds, target_names=["Non-requirement", "Requirement"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
