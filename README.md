# SRS Requirement Intelligence Suite

Professional UI/UX + backend for the three-model minor project pipeline:

1. **Model 1** extracts requirement sentences from a full SRS document.
2. **Model 2** classifies each requirement as Functional or Non-functional.
3. **Model 3** classifies Non-functional requirements into Availability, Performance, Security, or Usability.

## Project Structure

```text
srs_requirement_suite/
  app/
    main.py
    pipeline.py
  static/
    index.html
    styles.css
    app.js
  scripts/
    train_model1.py
    train_model2_fallback.py
    train_model3.py
  models/
    model1_requirement_detector.joblib
    FR_requirement_classifier_bert_tiny/
  requirements.txt
  README.md
```

## Model Sources

The app loads the models in this order:

```text
Model 1: models/model1_requirement_detector.joblib
Model 2: models/FR_requirement_classifier_bert_tiny/
Model 3: Hugging Face repo NISH7732/nfr-classifier
```

Model 1 extracts requirement-like sentences from the SRS document. Model 2 classifies extracted requirements as Functional or Non-functional. Model 3 runs only for Non-functional requirements and classifies them as Availability, Performance, Security, or Usability.

## Setup

```powershell
cd C:\Users\ISHVAJEET\Documents\Codex\2026-05-03\files-mentioned-by-the-user-best\srs_requirement_suite
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Website

```powershell
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Optional: Rebuild Local Artifacts

```powershell
python scripts\train_model1.py
python scripts\train_model2_fallback.py
python scripts\train_model3.py
```

The website uses the saved Model 1 and Model 2 artifacts by default. The fallback training scripts are kept for reproducibility.

## Environment Overrides

You can point the app at different model files or repositories with environment variables:

```powershell
$env:MODEL1_ARTIFACT="models\model1_requirement_detector.joblib"
$env:MODEL2_DIR="models\FR_requirement_classifier_bert_tiny"
$env:MODEL3_REPO_ID="NISH7732/nfr-classifier"
$env:MODEL3_LOCAL_ONLY="0"
$env:MODEL3_ARTIFACT="models\model3_nfr_type_classifier.joblib"
```

## Model 3: NFR Subclass Classification

This model is developed using DistilBERT and hosted on Hugging Face:

https://huggingface.co/NISH7732/nfr-classifier

Developed by:
Nishkarsh Gupta

Usage:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("NISH7732/nfr-classifier")
tokenizer = AutoTokenizer.from_pretrained("NISH7732/nfr-classifier")
```
