# 📚 Document Research Agent

<p align="center">
  <img src="assets/rag-logo.svg" alt="Document Research Agent logo" width="620">
</p>

<p align="center">
  <a href="https://github.com/GANESHPARDESHI07/RAG/actions/workflows/ci.yml">
    <img src="https://github.com/GANESHPARDESHI07/RAG/actions/workflows/ci.yml/badge.svg" alt="CI status">
  </a>
</p>

A production-oriented Retrieval-Augmented Generation (RAG) application for
chatting with PDF documents. It combines dense and sparse search, stores
document vectors in Qdrant, and uses LangGraph to decide whether a question
should be answered directly or answered with document search.

The application has a simple Streamlit interface for uploading PDFs, indexing
documents, viewing the number of indexed vectors, and asking questions in a
chat.

> **Get started:** Follow the [Installation Guide](INSTALLATION_GUIDE.md) to
> clone, configure, and run the project locally.

## ✨ Capabilities

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

## 🧠 Why This RAG Setup Is Effective

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

## 🧰 Technology and Specifications

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

## 📁 Repository Structure

```text
RAG/
├── app.py                         Streamlit application
├── RAG_cleared_cell_output.ipynb  RAG pipeline and LangGraph runtime
├── RAG.env.example                Configuration template
├── INSTALLATION_GUIDE.md          Local setup and usage instructions
├── assets/
│   └── rag-logo.svg               Project logo
├── qdrant-server/
│   └── docker-compose.yml         Local Qdrant service
├── .github/workflows/ci.yml       Automated validation
├── README.md                      Project overview
├── pyproject.toml                 Python dependencies
└── .gitignore                     Excludes secrets and generated data
```

`RAG.env` is intentionally not stored in Git.

## 🔒 Security and GitHub Readiness

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

## 🚀 Production Deployment Notes

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
