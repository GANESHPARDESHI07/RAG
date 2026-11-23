# Document Research Agent

A production-oriented Retrieval-Augmented Generation (RAG) application for
chatting with PDF documents. It combines dense and sparse search, stores
document vectors in Qdrant, and uses LangGraph to decide whether a question
should be answered directly or answered with document search.

The application has a simple Streamlit interface for uploading PDFs, indexing
documents, viewing the number of indexed vectors, and asking questions in a
chat.

## What This Agent Can Do

- Upload and index one or more PDF documents.
- Avoid duplicate indexing by creating a stable ID from the file name and
  document content.
- Search documents using both meaning and exact keywords.
- Preserve page numbers and document names in search results.
- Answer questions using relevant content from the indexed documents.
- List the documents and document statistics already stored in Qdrant.
- Keep chat history within the active Streamlit session.
- Send optional tracing information to Langfuse.
- Reject encrypted PDFs, unreadable PDFs, PDFs with no extractable text,
  oversized PDFs, and oversized uploads.

This system is useful for:

- Company policy and compliance documents
- Research reports and technical manuals
- Financial or legal documents
- Long PDF collections that are difficult to search manually
- Internal knowledge bases
- Question-and-answer tools for document-based work

## Why This RAG Setup Is Effective

### Hybrid retrieval

The agent uses both:

- **Dense retrieval:** FastEmbed creates meaning-based vectors.
- **Sparse retrieval:** BM25-style sparse vectors find exact keywords.

Qdrant combines both result sets with Reciprocal Rank Fusion (RRF). This helps
the system find documents that are relevant by meaning and documents that
contain exact names, numbers, section titles, or technical terms.

### Context-aware chunks

PDF text is split into chunks of 1,000 characters with 150 characters of
overlap. Every chunk receives contextual information about the document,
section, and source page before it is indexed. The overlap reduces the chance
of losing an answer because its text crossed a chunk boundary.

### Stable and repeatable indexing

Each document receives a content-based title. Re-uploading the same PDF does
not create duplicate points. Qdrant point IDs are also generated from stable
document and page information.

### Controlled agent behavior

LangGraph routes the conversation through separate router and agent nodes.
The router decides whether the question needs:

- Document retrieval
- Listing indexed documents
- A direct answer

The agent is given two controlled tools:

- `search_documents`
- `list_indexed_documents`

This limits the agent to the operations required for document research.

### External vector storage

Qdrant runs as a separate service. Its data remains available independently
of the Streamlit process. The included Docker setup keeps Qdrant's API ports
bound to the local machine.

## Technology and Specifications

| Component | Specification |
|---|---|
| Application interface | Streamlit |
| Agent framework | LangGraph |
| LLM integration | LangChain OpenAI-compatible chat client |
| Vector database | Qdrant |
| PDF extraction | PyMuPDF |
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse embeddings | `Qdrant/bm25` |
| Fusion method | Reciprocal Rank Fusion |
| Chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Default retrieval size | 5 results |
| Default indexing batch size | 32 points |
| Maximum PDF size | 50 MB by default |
| Maximum PDF pages | 2,000 by default |
| Maximum total upload size | 200 MB by default |
| Required Python | Python 3.12 |
| Optional tracing | Langfuse |

These defaults can be changed through environment variables.

## Repository Files

```text
RAG/
├── app.py                         Streamlit application
├── RAG_cleared_cell_output.ipynb  RAG pipeline and LangGraph runtime
├── RAG.env.example                Configuration template
├── qdrant-server/
│   └── docker-compose.yml         Local Qdrant service
├── pyproject.toml                 Python dependencies
└── .gitignore                     Excludes secrets and generated data
```

`RAG.env` is intentionally not stored in Git.

## Requirements

Install these before setup:

- Python 3.12
- Docker with Docker Compose
- An OpenAI-compatible LLM endpoint
- An API key for that endpoint
- About 2 GB of free disk space for Python packages and embedding models
- Internet access when downloading Python packages or embedding models

Python packages are installed automatically from `pyproject.toml`.

## Quick Setup

Run every command from the `RAG` folder.

### 1. Create the environment file

On Linux or macOS:

```bash
cp RAG.env.example RAG.env
```

On Windows PowerShell:

```powershell
Copy-Item RAG.env.example RAG.env
```

Open `RAG.env` and replace the placeholder values. Keep the real API key only
in this local file.

### 2. Configure the PDF

Set `PDF_PATH` to the absolute path of the PDF you want to index:

```dotenv
PDF_PATH=/absolute/path/to/document.pdf
```

The application can also upload PDFs through the browser, but the notebook
runtime still requires `PDF_PATH` during initialization.

### 3. Configure Qdrant

For the included local Docker setup, use:

```dotenv
QDRANT_URL=http://127.0.0.1:6333
QDRANT_TIMEOUT=30
COLLECTION_NAME=Agentic_rag
```

Start Qdrant:

```bash
docker compose -f qdrant-server/docker-compose.yml up -d
```

Confirm that Qdrant is running:

```bash
curl http://127.0.0.1:6333
```

Stop Qdrant when it is no longer needed:

```bash
docker compose -f qdrant-server/docker-compose.yml down
```

### 4. Configure the language model

Set the variables for any OpenAI-compatible API:

```dotenv
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-secret-api-key
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2
```

For a provider that follows the OpenAI API format, `LLM_BASE_URL` should
normally end in `/v1`. Do not publish `LLM_API_KEY` in GitHub commits, issues,
screenshots, or notebooks.

### 5. Create the Python environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### 6. Start the application

```bash
streamlit run app.py
```

Open the URL printed in the terminal, usually:

```text
http://localhost:8501
```

## Using the Application

1. Wait for the Qdrant connection and RAG runtime to initialize.
2. Upload PDF documents in the left sidebar.
3. Select **Index documents**.
4. Use the chat box to ask questions.
5. Ask for a list of indexed documents if needed.

Example questions:

- “What are the main recommendations in this report?”
- “Summarize chapter 3.”
- “What is the document saying about cost and timelines?”
- “Which indexed documents mention blockchain?”
- “List all indexed documents.”
- “What page contains the security requirements?”

The upload limit and page limit are controlled by:

```dotenv
RAG_MAX_FILE_SIZE_MB=50
RAG_MAX_PDF_PAGES=2000
RAG_MAX_TOTAL_UPLOAD_MB=200
```

## Environment Variables

| Variable | Required | Purpose |
|---|---:|---|
| `PDF_PATH` | Yes | Absolute path to the initial PDF file |
| `QDRANT_URL` | Yes | Qdrant HTTP endpoint |
| `QDRANT_TIMEOUT` | Yes | Qdrant request timeout in seconds |
| `COLLECTION_NAME` | Yes | Qdrant collection name |
| `DENSE_MODEL_NAME` | Yes | FastEmbed dense model name |
| `SPARSE_MODEL_NAME` | Yes | FastEmbed sparse model name |
| `LLM_BASE_URL` | Yes | OpenAI-compatible provider base URL |
| `LLM_MODEL` | Yes | Model available from the provider |
| `LLM_API_KEY` | Yes | Provider API key |
| `LLM_TIMEOUT` | Yes | LLM request timeout |
| `LLM_MAX_RETRIES` | Yes | LLM retry count |
| `TOP_K` | Yes | Number of search results |
| `BATCH_SIZE` | Yes | Qdrant indexing batch size |
| `LANGFUSE_PUBLIC_KEY` | No | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | Langfuse secret key |
| `LANGFUSE_HOST` | No | Langfuse server URL |
| `RAG_MAX_FILE_SIZE_MB` | No | Maximum size of one PDF |
| `RAG_MAX_PDF_PAGES` | No | Maximum pages in one PDF |
| `RAG_MAX_TOTAL_UPLOAD_MB` | No | Maximum combined upload size |

## Optional Langfuse Tracing

To enable tracing, add your Langfuse credentials to `RAG.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Leave these values empty if tracing is not needed.

## PDF Requirements

The application works best with PDFs that contain normal selectable text.

Supported:

- Text-based PDF reports
- Searchable PDF manuals
- Documents with consistent headings

Not supported:

- Password-protected PDFs
- Scanned images without OCR
- PDFs with no extractable text
- Files larger than configured limits

For scanned documents, run OCR on them before uploading.

## Security and GitHub Readiness

Before publishing or deploying:

- Do not commit `RAG.env`.
- Do not commit API keys or private model credentials.
- Do not commit PDF data or generated Qdrant storage.
- Use environment variables or a secrets manager in production.
- Keep Qdrant private or place it behind authentication and a firewall.
- Restrict the Streamlit application and its network access.
- Use HTTPS for public deployments.
- Review uploaded documents for personal, legal, or confidential information.
- Replace the default embedding and model names with approved organization
  models when required.

The `.gitignore` file excludes local environment files, uploaded PDFs, Python
caches, checkpoints, Qdrant storage, SQLite files, and other generated data.

## Production Deployment Notes

The included Docker Compose file is suitable for local or private development.
For a public production deployment, also consider:

- Using an authentication proxy or SSO in front of Streamlit
- Running Qdrant in an authenticated managed or private environment
- Using a persistent and encrypted volume for Qdrant
- Setting resource limits and health checks
- Using HTTPS and a reverse proxy
- Pinning the Qdrant image version instead of using `latest`
- Running the application behind a process supervisor or container platform
- Rotating API keys regularly
- Keeping uploaded documents in encrypted storage
- Adding document retention and access-control policies
- Backing up the Qdrant collection
- Logging application events without recording API keys or document content

Do not expose Qdrant's ports directly to the public internet.

## Troubleshooting

### The application says `RAG.env` is missing

Create it from the template:

```bash
cp RAG.env.example RAG.env
```

Then fill in the required values.

### Qdrant will not connect

Check that Docker is running:

```bash
docker compose -f qdrant-server/docker-compose.yml ps
docker compose -f qdrant-server/docker-compose.yml logs qdrant
```

Make sure `QDRANT_URL` matches `http://127.0.0.1:6333` for the local setup.

### The configured collection is missing

Create and index the collection by running the required setup cells in
`RAG_cleared_cell_output.ipynb`, or prepare the collection using the same
`COLLECTION_NAME` value as `RAG.env`.

### The PDF cannot be read

Check that `PDF_PATH` is an absolute path and that the file exists. Also check
whether the PDF is encrypted or consists only of scanned images.

### The LLM initialization fails

Verify all of these values:

- `LLM_BASE_URL`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_TIMEOUT`
- `LLM_MAX_RETRIES`

The provider may require a model name different from the example.

### Module or dependency errors

Activate the virtual environment and reinstall dependencies:

```bash
source .venv/bin/activate
python -m pip install -e .
```

## Validation

Compile the application:

```bash
python -m py_compile app.py
```

Validate the notebook JSON:

```bash
python - <<'PY'
import json

with open("RAG_cleared_cell_output.ipynb", encoding="utf-8") as file:
    json.load(file)

print("Notebook JSON: OK")
PY
```

Start the web interface:

```bash
streamlit run app.py
```
