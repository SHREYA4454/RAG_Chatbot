# Open PDF RAG

A locally hosted PDF question-answering demo. It uses PyMuPDF for native text,
Tesseract OCR when a page has little extractable text and for embedded raster
images on text-bearing pages, deterministic language identification, a free
Sentence Transformers embedding model, and Chroma's persistent HNSW index. Ollama
provides local generation; the interface always displays retrieved PDF/page
citations and clearly marks when it has fallen back to an extract rather than
an LLM-generated answer.

For the full architecture, corpus inventory, API, evaluation, and current
validation status, see [PROJECT_GUIDE.pdf](PROJECT_GUIDE.pdf) or its editable
source [PROJECT_GUIDE.md](PROJECT_GUIDE.md). Regenerate the PDF with
`python generate_project_guide.py`.

## Supplied astronomy corpus and measured status

The supplied corpus contains 10 PDFs, 3,900 pages total. Every book exceeds
the 200-page minimum. PyMuPDF's native-text check found fewer than 40
non-whitespace characters on 2,270 pages; those pages will use full-page OCR.
For text-bearing pages, 2,978 of 2,983 image placements cover at least 70% of
the page and are skipped as duplicate OCR; small embedded images may still be
OCRed. The source files have been copied to `data/` for ingestion. These are input
files, not an already-built vector index: ingestion has not yet been run, so
the indexed-passage count and answer latency are not measured. Run the steps
below to build the local index.

| PDF | Pages |
|---|---:|
| `01_Introduction_to_Astronomy.pdf` | 601 |
| `02_Romance_of_Modern_Astronomy.pdf` | 424 |
| `03_Treatise_for_Colleges.pdf` | 339 |
| `04_Theoretical_and_Descriptive_Astronomy.pdf` | 284 |
| `05_Elements_of_Astronomy_Rolfe.pdf` | 380 |
| `06_Story_of_the_Stars.pdf` | 356 |
| `07_Compendium_of_Astronomy.pdf` | 317 |
| `08_Navigation_and_Nautical_Astronomy.pdf` | 240 |
| `09_Astronomy_of_the_Bible.pdf` | 442 |
| `10_Elements_of_Astronomy_Young.pdf` | 517 |
| **Total** | **3,900** |

The 2–5 second response-time objective depends on hardware, model startup,
corpus, and local Ollama performance. Use the live p95 metric and benchmark
script; this project does not claim a latency guarantee before measurement.

## Quick start (Docker)

Requirements: Docker Compose and enough memory/storage for the embedding model,
the local LLM, and your PDF index.

Docker Desktop / the Docker Engine must be running. The supplied PDFs are
already staged in `data/`; for a different corpus, place PDFs there (subfolders
are scanned recursively).

1. Start the stack:

   ```powershell
   docker compose up --build -d
   ```

2. Download the default local answer model (first run only):

   ```powershell
   docker compose exec ollama ollama pull qwen2.5:3b
   ```

3. Open <http://localhost:8000>, choose **Index PDFs in ./data**, and wait for
   ingestion to finish. This corpus includes many scanned pages; OCR and
   embedding may take substantial time. The first embedding-model load also
   downloads `sentence-transformers/all-MiniLM-L6-v2`.
4. Ask a question. The answer includes source excerpts with filename, page,
   similarity score, and response mode.

The vector index and manifest persist in `vector_db/`; downloaded Ollama models
persist in a Docker volume. The `data/` directory is bind-mounted. The initial
embedding model download is cached in the container layer/runtime and may
re-download if the container is rebuilt.

## Run without Docker

Use Python 3.11+, install Tesseract OCR (including the language data for
`OCR_LANG`), install the packages from `requirements.txt`, and install/run
Ollama with the configured model. Then:

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
uvicorn app:app --reload
```

By default, PDFs are read from `./data` and the index is stored in
`./vector_db`. Environment variables can override these paths and model names.
For OCR, scanned pages or pages with fewer than 40 non-whitespace extracted
characters are rendered and sent through Tesseract. On text-bearing pages,
embedded raster images larger than 30,000 pixels are OCRed separately. Install
additional OCR language packs and set `OCR_LANG` for non-English documents.

## API

- `GET /api/health` — liveness check.
- `GET /api/status` — corpus coverage, ingestion status, and index size.
- `POST /api/ingest` — asynchronously index all PDFs under `RAG_DATA_DIR`.
- `POST /api/chat` — `{"question":"...", "top_k":5}`.
- `GET /api/metrics` — in-process query count, mean/p95 latency, and fallback
  count (metrics reset when the service restarts).

Chunking is deterministic and page-scoped: 650 whitespace-delimited words per
passage with 130 words of overlap. Each indexed passage stores filename, page,
detected language, source identifier, file hash, and chunk number. Re-ingesting
a file replaces its prior passages. Extraction removes lines repeated across
pages after page text extraction when those lines recur in page-edge blocks;
extraction order and page boundaries are preserved.

## Evaluation and tuning

Create a UTF-8 JSON file with labeled questions and relevant filenames:

```json
[
  {"query": "What is the policy for ...?", "relevant": ["policy.pdf"]},
  {"query": "When was ...?", "relevant": ["report-a.pdf", "report-b.pdf"]}
]
```

Run `python evaluate.py benchmark.json` after ingestion. It reports filename
Recall@k, MRR, and end-to-end mean/p95 latency. These are dataset-specific
measurements, not a guarantee of answer correctness or citation accuracy.

For lower latency, use a GPU for embeddings and Ollama, warm the models before
measurement, choose a smaller Ollama model that fits your hardware, and tune
`top_k`. Preserve the source panel and test answer quality after changing
retrieval settings. The embedding model and vector database are open-source;
Ollama and the chosen generation model run locally.

## Notes

- The demo is intended for local/private use. Do not expose it to untrusted
  networks without adding authentication, authorization, rate limits, and
  deployment hardening.
- The source PDFs in `Astronomy_RAG_PDFs/` and `data/`, and generated vector
  database files in `vector_db/`, are excluded by `.gitignore`; do not commit
  private corpus files or the generated index to a public repository.
- The ingestion endpoint indexes local files already placed in the configured
  data directory; it does not accept arbitrary filesystem paths from API
  callers.
- Page citations refer to the PDF's 1-based page number. OCR bounding boxes
  are not currently exposed.
