# 📄 Digitalizer: Offline-First OCR Pipeline & Academic Knowledge Enrichment

[![Language](https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Vision OCR](https://img.shields.io/badge/Vision%20OCR-GLM--4V%20%7C%20PyMuPDF-green)](#)
[![Local LLM](https://img.shields.io/badge/Local%20LLM-Ollama%20(qwen2.5)-orange)](#)
[![Cloud Fallback](https://img.shields.io/badge/Enrichment-Google%20Gemini-blue)](#)
[![Integration](https://img.shields.io/badge/Sync-Notion%20API-black?logo=notion)](#)

> Offline-first visual OCR pipeline powered by local vision models, two-tier disk caching, and semantic academic enrichment for Notion knowledge databases.

---

## 📌 Executive Summary

Converting academic manuscripts, handwritten lecture notes, and dense technical PDFs into structured knowledge typically causes unreadable raw text dumps and expensive cloud API bills.

Digitalizer provides a two-stage extraction architecture: local vision OCR (`glm-4v` / `qwen2.5` via Ollama) followed by structured academic enrichment.

The pipeline incorporates a two-tier persistent caching mechanism to eliminate redundant vision computation, uploading deduplicated, topic-clustered notes into Notion.

---

## 🏛️ Two-Tier Pipeline Architecture

```
                          DIGITALIZER PIPELINE
                                   │
                     Dense Technical Document (PDF)
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │             PHASE 1: EXTRACTION & NORMALIZATION       │
       │       PyMuPDF Rendering → Vision OCR (GLM-4V/Qwen)    │
       │       Two-Tier Cache (raw_cache/ → reviewed_cache/)   │
       └───────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │          PHASE 2: SEMANTIC MAPPING & BATCH UPLOAD     │
       │       Cross-Batch Topic Clustering (ocr_pipeline.py)  │
       │       Chunked Notion Database Ingestion (10 pages)    │
       └───────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
       ┌───────────────────────────────────────────────────────┐
       │         PHASE 3: ACADEMIC NOTE ENRICHMENT (OPTIONAL)  │
       │       Gemini University Professor Prompt Engine       │
       │       Syntax Expansion, Definitions (enrich_notes.py) │
       └───────────────────────────────────────────────────────┘
```

The offline-first foundation guarantees zero cloud API costs during intensive PDF extraction while maintaining complete privacy over source materials.

---

## ⚙️ Core Technical Modules

### Verified Pipeline Architecture
- **Visual OCR & Clustering Core ([`ocr_pipeline.py`](ocr_pipeline.py)):** Extracts PDF pages via PyMuPDF, executes visual OCR, manages two-tier disk caches, clusters topics semantically, and handles batch Notion uploads.
- **Academic Enrichment Engine ([`enrich_notes.py`](enrich_notes.py)):** Pulls staged OCR drafts from Notion and applies deep academic structuring: expanding abbreviations, formatting formulas, and injecting conceptual blockquotes.

---

## 🛠️ Production Quickstart

### 1. Environment Configuration
Install dependencies and ensure local Ollama vision models are pulled:
```powershell
# Install Python packages
# Dependencies include: pymupdf, ollama, requests, python-dotenv, rich
pip install pymupdf ollama requests python-dotenv rich

# Pull local OCR post-processing model
ollama pull qwen2.5:14b
```

### 2. End-to-End Execution
Run document ingestion or trigger academic knowledge enrichment:
```powershell
# Execute visual OCR extraction, semantic clustering, and Notion upload
python ocr_pipeline.py

# Enrich existing Notion notes with academic explanations and definitions
python enrich_notes.py
```

---

**Author:** Francesco Colombini  
[GitHub Profile](https://github.com/FRA-0023) · [LinkedIn](https://www.linkedin.com/in/francescocolombini/)