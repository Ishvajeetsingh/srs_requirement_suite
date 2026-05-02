from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import RequirementPipeline, extract_text_from_upload


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "static"

app = FastAPI(title="SRS Requirement Intelligence Suite", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RequirementPipeline()

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return pipeline.status()


@app.post("/api/analyze")
async def analyze(text: str = Form(""), file: UploadFile | None = File(None)):
    source_text = text.strip()
    if file and file.filename:
        content = await file.read()
        try:
            source_text = extract_text_from_upload(file.filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not source_text.strip():
        raise HTTPException(status_code=400, detail="Paste SRS text or upload a TXT, PDF, or DOCX file.")
    return pipeline.analyze(source_text)
