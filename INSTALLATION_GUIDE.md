# Installation Guide

Follow these steps to run the Document Research Agent on your computer.

## 1. Requirements

Install or prepare:

- Python 3.12
- Git
- Docker with Docker Compose
- An OpenAI-compatible LLM endpoint and API key
- Internet access to install packages and download embedding models

Allow about 2 GB of free disk space for Python packages and embedding models.

## 2. Clone the repository

```bash
git clone https://github.com/GANESHPARDESHI07/RAG.git
cd RAG
```

## 3. Create and configure `RAG.env`

Linux or macOS:

```bash
cp RAG.env.example RAG.env
```

Windows PowerShell:

```powershell
Copy-Item RAG.env.example RAG.env
```

Open `RAG.env` and set the values for your machine and service providers.
At minimum, configure:

```dotenv
PDF_PATH=/absolute/path/to/your/document.pdf

QDRANT_URL=http://127.0.0.1:6333
QDRANT_TIMEOUT=30
COLLECTION_NAME=Agentic_rag

DENSE_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
SPARSE_MODEL_NAME=Qdrant/bm25

LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-secret-api-key
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2

TOP_K=5
BATCH_SIZE=32
```

Use the full path to a PDF that exists on your computer. The application also
supports uploading PDFs through the interface, but `PDF_PATH` is still needed
when the RAG runtime starts.

Keep API keys in `RAG.env`. Never commit this file or paste secrets into the
notebook.

## 4. Start Qdrant

Start the local Qdrant service from the repository folder:

```bash
docker compose -f qdrant-server/docker-compose.yml up -d
```

Check that Qdrant is responding:

```bash
curl http://127.0.0.1:6333
```

The included Docker configuration binds Qdrant to localhost. Do not expose
these ports directly to the public internet.

## 5. Create a Python environment and install dependencies

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The first application start may take longer while FastEmbed downloads the
configured embedding models.

## 6. Create the Qdrant collection

The app expects a Qdrant collection to exist before it starts; it does not
create collections automatically. The configured collection must contain a
named dense vector called `dense` (384 dimensions for the default
`all-MiniLM-L6-v2` model) and a named sparse vector called `sparse`.

For a new local collection, activate the virtual environment and run:

```bash
python - <<'PY'
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv("RAG.env")

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    timeout=int(os.environ["QDRANT_TIMEOUT"]),
)
collection_name = os.environ["COLLECTION_NAME"]

if client.collection_exists(collection_name):
    print(f"Collection already exists: {collection_name}")
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    print(f"Created collection: {collection_name}")
PY
```

If you use an existing collection or a different dense embedding model, make
sure its vector names and dimensions match the configured models and the
notebook's search implementation.

## 7. Run the application

Linux or macOS:

```bash
source .venv/bin/activate
streamlit run app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
[`http://localhost:8501`](http://localhost:8501).

## 8. Index and search documents

1. Wait for the app to connect to Qdrant and initialize the RAG runtime.
2. Upload one or more text-based PDFs in the sidebar.
3. Select **Index documents**.
4. Ask a question in the chat box.
5. Ask the agent to list the documents currently indexed if needed.

Example questions:

- “Summarize the main recommendations in this report.”
- “What does the document say about cost and timelines?”
- “Which pages mention security requirements?”
- “List the indexed documents.”

The app rejects encrypted PDFs, unreadable PDFs, files with no extractable
text, and uploads over the configured size or page limits. Scanned PDFs need
OCR before they can be searched.

## Optional: Langfuse tracing

To enable Langfuse, set these values in `RAG.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Leave the keys blank to run without optional tracing.

## Configuration reference

| Variable | Purpose |
|---|---|
| `PDF_PATH` | Absolute path to a local PDF used during runtime initialization |
| `QDRANT_URL` | Qdrant HTTP endpoint |
| `QDRANT_TIMEOUT` | Qdrant request timeout in seconds |
| `COLLECTION_NAME` | Qdrant collection to use |
| `DENSE_MODEL_NAME` | FastEmbed dense embedding model |
| `SPARSE_MODEL_NAME` | FastEmbed sparse embedding model |
| `LLM_BASE_URL` | OpenAI-compatible API base URL |
| `LLM_MODEL` | Model name supported by the provider |
| `LLM_API_KEY` | Secret key for the LLM provider |
| `LLM_TIMEOUT` | LLM request timeout |
| `LLM_MAX_RETRIES` | Number of LLM request retries |
| `TOP_K` | Number of search results |
| `BATCH_SIZE` | Number of points sent per indexing batch |
| `LANGFUSE_PUBLIC_KEY` | Optional Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Optional Langfuse secret key |
| `LANGFUSE_HOST` | Optional Langfuse server URL |
| `RAG_MAX_FILE_SIZE_MB` | Maximum size of one PDF (default: 50 MB) |
| `RAG_MAX_PDF_PAGES` | Maximum pages per PDF (default: 2,000) |
| `RAG_MAX_TOTAL_UPLOAD_MB` | Maximum combined upload size (default: 200 MB) |

## Troubleshooting

### `RAG.env` is missing

Copy `RAG.env.example` to `RAG.env` and fill in the required values.

### Qdrant does not connect

Check Docker and the Qdrant logs:

```bash
docker compose -f qdrant-server/docker-compose.yml ps
docker compose -f qdrant-server/docker-compose.yml logs qdrant
```

Confirm `QDRANT_URL` is `http://127.0.0.1:6333` for the included local setup.

### The collection is missing or incompatible

Confirm that the collection named in `COLLECTION_NAME` exists and has dense
and sparse vectors named `dense` and `sparse`. The dense vector size must
match the selected embedding model.

### The PDF cannot be read

Confirm `PDF_PATH` is an absolute path and that the file exists. Password
protection and image-only scans without OCR are not supported.

### LLM initialization fails

Check `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, and the provider's API
compatibility. Never include your key in a support request.

### Python dependency errors

Make sure the virtual environment is active, then run:

```bash
python -m pip install -e .
```

## Stop local services

Stop Streamlit with `Ctrl+C`. Stop Qdrant with:

```bash
docker compose -f qdrant-server/docker-compose.yml down
```

Qdrant data is stored locally under `qdrant-server/storage/` and is ignored by
Git.
