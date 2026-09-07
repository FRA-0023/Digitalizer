# Digitalizer: Offline-First Multimodal OCR & Semantic Knowledge Engine

[![Language](https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![Vision OCR](https://img.shields.io/badge/Vision%20OCR-GLM--4V%20%7C%20Qwen2.5--VL-green)](#)
[![Local LLM](https://img.shields.io/badge/Local%20LLM-Ollama%20(qwen2.5%3A14b)-orange)](#)
[![Knowledge Base](https://img.shields.io/badge/Knowledge%20Base-Notion%20API%20(AST%20Parser)-black?logo=notion)](#)
[![Data Privacy](https://img.shields.io/badge/Privacy-100%25%20Local%20Inference-brightgreen)](#)

> Offline-first visual document extraction engine bridging local multimodal vision models, two-tier persistent disk caching, and hierarchical Notion knowledge synchronization.

---

## 📌 Executive Summary

Digitizing dense technical manuscripts, mathematical lecture notes, and multi-column research papers typically imposes an unacceptable dilemma: surrender confidential documents to cloud OCR APIs at compounding per-page costs, or accept legacy OCR engines (e.g., Tesseract) that mutilate mathematical notation and structural layout.

Digitalizer resolves this operational bottleneck with an offline-first multimodal pipeline executing 100% locally via Ollama. 

High-resolution PDF rasterization feeds local vision models (`glm-4v`, `qwen2.5-vl`), isolated behind a two-tier persistent caching architecture that eliminates redundant GPU compute and ingests structured AST blocks directly into Notion.

---

## 🏛️ Pipeline Architecture

```
                       DIGITALIZER ENGINE
                               │
               Dense Technical Document (PDF)
                               │
                               ▼
    ┌─────────────────────────────────────────────────────┐
    │          PHASE 1: LOCAL INGESTION & VISION OCR      │
    │  PyMuPDF High-Res Rendering (300 DPI Pixmaps)       │
    │  Local Multimodal Inference via Ollama (GLM-4V)     │
    │  Tier 1 Storage: cache_ocr_grezzo/ (Raw Transcripts)│
    └──────────────────────────┬──────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────┐
    │     PHASE 2: NORMALIZATION & SEMANTIC CLUSTERING    │
    │  Formula-Preserving Heuristic Case Normalization    │
    │  Regex Lecture & Chapter Boundary Identification    │
    │  Semantic Topic Map via Local Qwen 2.5 (14B)        │
    │  Tier 2 Storage: cache_ocr_revisionato/ (Clean AST) │
    └──────────────────────────┬──────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────┐
    │     PHASE 3: AST BLOCK PARSING & NOTION SYNC        │
    │  Native Markdown-to-Notion AST Block Generator      │
    │  Payload Chunking (Max 100 Blocks / 2000 Chars)     │
    │  Exponential Backoff Retries with Jitter            │
    └──────────────────────────┬──────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────┐
    │     PHASE 4: ACADEMIC NOTE ENRICHMENT (OPTIONAL)    │
    │  Professor Prompt Engine (Gemini / Local LLM)       │
    │  Acronym Expansion, Formula LaTeX & Key Takeaways   │
    │  Atomic Block Stream Replacement in Knowledge Base  │
    └─────────────────────────────────────────────────────┘
```

The offline-first foundation guarantees zero API expenditure during intensive raster extraction while ensuring complete data confidentiality over unpublished intellectual property.

---

## 🔬 Core Technical Modules

### 1. Multimodal Vision OCR & Caching Core ([`ocr_pipeline.py`](ocr_pipeline.py))
- **High-Fidelity Rasterization:** Employs PyMuPDF to extract raw page vectors and render them as optimized pixmaps, preserving high-frequency text boundaries and dense diagrams.
- **Two-Tier Disk Caching:** Isolates raw transcriptions in `cache_ocr_grezzo/` before feeding them into normalization. Polished, structured outputs are committed to `cache_ocr_revisionato/`, preventing duplicate vision inference on unchanged source documents.
- **Structural Case Normalization:** Implements specialized heuristics (`_correct_case_python`) that correct OCR capitalization artifacts without altering mathematical notation, variable names, or programmatic code snippets.
- **Semantic Topic Clustering:** Deploys `qwen2.5:14b` locally to infer macro-thematic boundaries across fragmented lecture slides, grouping loose pages into coherent hierarchical chapters.

### 2. Native AST Markdown-to-Notion Synchronizer ([`ocr_pipeline.py`](ocr_pipeline.py))
- **Block-Level Translation:** Recursively converts raw Markdown into native Notion block schemas (`heading_1`, `heading_2`, `heading_3`, `code`, `callout`, `bulleted_list_item`, `equation`).
- **Hard Limit Chunking:** Enforces Notion API specifications, dynamically splitting text buffers exceeding 2,000 characters and batching payloads into strict 100-block atomic operations.
- **Fault-Tolerant Transport:** Implements exponential backoff with randomized jitter (`@retry(retries=3, delay=2.0)`), defending ingestion jobs against transient network drops and HTTP 429 rate limits.

### 3. Academic Knowledge Enrichment Engine ([`enrich_notes.py`](enrich_notes.py))
- **Professor-Level Structuring:** Connects to high-speed LLM inference engines (Google Gemini or local fallbacks) to expand cryptic lecture shorthand into rigorous academic prose.
- **LaTeX Normalization:** Detects raw mathematical expressions and converts them into standardized LaTeX equations for native display.
- **Atomic Block Replacement:** Flushes preliminary OCR placeholder blocks (`clear_blocks`) and streams enriched conceptual notes with zero database corruption or orphan records.

---

## 📊 Architectural & Economic Trade-Off Matrix

| Capability | Digitalizer (Local Engine) | Cloud Vision APIs (AWS / GCP) | Legacy Rule-Based (Tesseract) |
|---|:---:|:---:|:---:|
| **Marginal Compute Cost** | **$0.00 (Local Hardware)** | $1.50–$2.50 per 1k pages | $0.00 |
| **Confidentiality / IP Defense** | **Zero Data Leakage (Air-Gapped)** | External cloud transit | Zero Data Leakage |
| **Mathematical / LaTeX Retention** | **High (Multimodal Context)** | Medium (Frequent formatting loss) | Poor (Unreadable character noise) |
| **Topic Clustering & Segmentation** | **Automated via Local LLM** | Manual post-processing required | None |
| **State Caching & Resumability** | **Native Two-Tier Filesystem** | Requires custom external cache | None |

---

## 🛠️ Environment & Production Quickstart

### 1. Prerequisites & Dependencies
Ensure Python 3.10+ and Ollama are installed and running locally:
```powershell
# Clone the repository
git clone https://github.com/FRA-0023/Digitalizer.git
cd Digitalizer

# Install required Python packages
pip install pymupdf ollama requests python-dotenv rich google-genai
```

### 2. Local Vision & Language Models
Pull the recommended vision and language models via Ollama:
```powershell
# Pull multimodal vision model
ollama pull glm-4v

# Pull semantic clustering model
ollama pull qwen2.5:14b
```

### 3. Environment Configuration
Configure credentials in `.env` (referencing [`.gitignore`](.gitignore)):
```env
NOTION_TOKEN="your_notion_integration_token"
NOTION_DATABASE_ID="your_notion_database_id"
GEMINI_API_KEY="your_gemini_api_key" # Optional, used exclusively for enrich_notes.py
```

### 4. End-to-End Execution
Run document ingestion or trigger academic knowledge enrichment:
```powershell
# Execute visual OCR extraction, two-tier caching, and Notion upload
python ocr_pipeline.py

# Enrich existing Notion notes with academic definitions and LaTeX equations
python enrich_notes.py
```

---

**Author:** Francesco Colombini  
[GitHub Profile](https://github.com/FRA-0023) · [LinkedIn](https://www.linkedin.com/in/francescocolombini/)
