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
    .gitkeep
  requirements.txt
  README.md
```

## Your Current Source Files

The app defaults to these files on your machine:

```text
Model 1 metadata: C:\Users\ISHVAJEET\Desktop\python\model_data.json
Model 1 dataset:  C:\Users\ISHVAJEET\Downloads\requirement_detection_final.csv

Model 2 folder:   C:\Users\ISHVAJEET\Downloads\FR_requirement_classifier_bert_tiny)
Model 2 dataset:  C:\Users\ISHVAJEET\Downloads\clean_requirements_dataset.csv

Model 3 notebook: C:\Users\ISHVAJEET\Downloads\model3.ipynb
Model 3 dataset:  C:\Users\ISHVAJEET\Downloads\model3_cleaned_datasetCopy.csv
```

Important: `model_data.json` contains the Model 1 TF-IDF vocabulary and metadata, but not the trained classifier weights. Model 3 has the notebook and dataset, but no saved model folder. Run the training scripts once to create the missing runnable artifacts.

## Setup

```powershell
cd C:\Users\ISHVAJEET\Documents\Codex\2026-05-03\files-mentioned-by-the-user-best\srs_requirement_suite
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build Missing Model Artifacts

```powershell
python scripts\train_model1.py
python scripts\train_model2_fallback.py
python scripts\train_model3.py
```

Model 2 will use your saved BERT-tiny folder directly when `transformers` and `torch` are installed. The fallback script is included so the website still works if those libraries are unavailable.

## Run The Website

```powershell
uvicorn app.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Environment Overrides

You can point the app at different files with environment variables:

```powershell
$env:MODEL1_ARTIFACT="models\model1_requirement_detector.joblib"
$env:MODEL2_DIR="C:\Users\ISHVAJEET\Downloads\FR_requirement_classifier_bert_tiny)"
$env:MODEL2_FALLBACK_ARTIFACT="models\model2_fr_nfr_fallback.joblib"
$env:MODEL3_ARTIFACT="models\model3_nfr_type_classifier.joblib"
```
