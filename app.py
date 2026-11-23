
"""
Document Research Agent
Streamlit interface for the existing notebook-based LangGraph RAG system.

Required files in the same directory:
    app.py
    RAG_cleared_cell_output.ipynb
    RAG.env

The notebook must be trusted because selected code cells are executed.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import re
import sys
import types
import typing
import uuid

from pathlib import Path
from typing import Any

import typing_extensions
import pymupdf
import streamlit as st

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

NOTEBOOK_PATH = PROJECT_DIR / "RAG_cleared_cell_output.ipynb"
ENV_PATH = PROJECT_DIR / "RAG.env"

MAX_FILE_SIZE_MB = int(
    os.getenv("RAG_MAX_FILE_SIZE_MB", "50")
)

MAX_PDF_PAGES = int(
    os.getenv("RAG_MAX_PDF_PAGES", "2000")
)

MAX_TOTAL_UPLOAD_MB = int(
    os.getenv("RAG_MAX_TOTAL_UPLOAD_MB", "200")
)

# Resolve NotRequired independently of the notebook.
try:
    NotRequired = typing.NotRequired
except AttributeError:
    NotRequired = typing_extensions.NotRequired


# Resolve required notebook cells by unique source markers, not numeric indexes.
# This remains stable when Markdown or unrelated code cells are inserted.
REQUIRED_CELL_MARKERS = [
    ("load RAG.env configuration", 'ENV_FILE = Path.cwd() / "RAG.env"'),
    ("initialize Qdrant and embedding models", "dense_model = TextEmbedding("),
    ("define hybrid_search", "def hybrid_search("),
    ("define contextualize_chunk", "def contextualize_chunk("),
    ("configure text splitter", "text_splitter = RecursiveCharacterTextSplitter("),
    ("define create_page_chunks", "def create_page_chunks("),
    ("define index_chunks", "def index_chunks("),
    ("define prepare_qdrant_points", "def prepare_qdrant_points("),
    ("define search_documents tool", "def search_documents("),
    ("define list_indexed_documents tool", "def list_indexed_documents("),
    ("register RAG tools", "rag_tools = ["),
    ("define system prompts", "ROUTER_SYSTEM_PROMPT ="),
    ("define RAGState", "class RAGState("),
    ("initialize ChatOpenAI", "llm = ChatOpenAI("),
    ("bind tools to LLM", "llm_with_rag_tools = llm.bind_tools("),
    ("define router_node", "def router_node("),
    ("define agent_node", "def agent_node("),
    ("define conditional router", "def route_after_router("),
    ("build and compile LangGraph", "graph_builder = StateGraph(RAGState)"),
]


def locate_required_cells(cells: list[dict]) -> list[tuple[int, dict, str]]:
    """Locate required code cells by content, tolerating inserted Markdown cells."""
    located = []
    normalized_sources = []

    for cell in cells:
        raw_source = cell.get("source", "")
        cell_source = (
            "".join(raw_source)
            if isinstance(raw_source, list)
            else str(raw_source)
        )
        normalized_sources.append((cell, cell_source))

    for description, marker in REQUIRED_CELL_MARKERS:
        matches = [
            (index, cell, cell_source)
            for index, (cell, cell_source) in enumerate(normalized_sources)
            if cell.get("cell_type") == "code" and marker in cell_source
        ]

        if len(matches) != 1:
            found = [index for index, _, _ in matches]
            raise RuntimeError(
                f"Could not uniquely locate notebook step '{description}' "
                f"using marker {marker!r}. Matches: {found or 'none'}. "
                "Check whether the notebook code has changed."
            )

        located.append(matches[0])

    indexes = [index for index, _, _ in located]
    if indexes != sorted(indexes):
        order = ", ".join(
            f"{description} (cell {index})"
            for (description, _), index in zip(REQUIRED_CELL_MARKERS, indexes)
        )
        raise RuntimeError(
            "Required notebook cells are no longer in dependency order: " + order
        )

    return located



# ============================================================
# 2. LOGGING AND STREAMLIT CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("rag.application")

st.set_page_config(
    page_title="Document Research Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. NOTEBOOK RUNTIME INITIALIZATION
# ============================================================

def create_runtime_namespace(
    notebook_path: str,
) -> tuple[types.ModuleType, dict[str, Any]]:
    """
    Create a real Python module for notebook execution.

    This is important because LangGraph and Python's typing
    utilities may resolve class annotations through sys.modules.
    """

    module_name = "rag_notebook_runtime"

    # Remove an old runtime module before rebuilding it.
    sys.modules.pop(module_name, None)

    runtime_module = types.ModuleType(module_name)

    runtime_module.__file__ = notebook_path
    runtime_module.__package__ = ""

    namespace = runtime_module.__dict__

    namespace.update(
        {
            "__name__": module_name,
            "__file__": notebook_path,
            "__package__": "",
            "NotRequired": NotRequired,
            "typing": typing,
            "typing_extensions": typing_extensions,
        }
    )

    # Critical: register the module before executing notebook code.
    sys.modules[module_name] = runtime_module

    return runtime_module, namespace


@st.cache_resource(show_spinner="Initializing RAG runtime...")
def load_agent_runtime(
    notebook_path: str,
    project_dir: str,
    notebook_mtime_ns: int,
    env_mtime_ns: int,
) -> dict[str, Any]:

    del notebook_mtime_ns, env_mtime_ns

    notebook_file = Path(notebook_path)
    env_file = Path(project_dir) / "RAG.env"

    if not notebook_file.is_file():
        raise FileNotFoundError(
            f"Notebook not found: {notebook_file}"
        )

    if not env_file.is_file():
        raise FileNotFoundError(
            f"Environment file not found: {env_file}"
        )

    # The notebook's first cell loads RAG.env from the current directory.
    os.chdir(project_dir)

    with notebook_file.open(
        "r",
        encoding="utf-8",
    ) as file:

        notebook = json.load(file)

    cells = notebook.get("cells", [])

    runtime_module, namespace = create_runtime_namespace(
        str(notebook_file)
    )

    logger.info("Created notebook runtime module.")

    required_cells = locate_required_cells(cells)

    for cell_number, cell, source in required_cells:

        # The hybrid-search cell contains an old hardcoded model assignment.
        # Keep the model configured in RAG.env as the single source of truth.
        source = re.sub(
            r"""(?m)^\\s*DENSE_MODEL_NAME\\s*=\\s*['"]sentence-transformers/all-MiniLM-L6-v2['"]\\s*$""",
            "# Dense model is loaded from RAG.env; do not override it here.",
            source,
        )

        if not source.strip():
            continue

        if source.lstrip().startswith(("%", "!")):
            raise RuntimeError(
                f"Notebook cell {cell_number} contains "
                "a notebook-only command."
            )

        try:

            # Ensure the symbol exists before every execution.
            namespace["NotRequired"] = NotRequired

            code = compile(
                source,
                f"{notebook_file}:cell-{cell_number}",
                "exec",
            )

            # Execute in the registered module's actual namespace.
            exec(
                code,
                namespace,
                namespace,
            )

            logger.info(
                "Successfully executed notebook cell %s",
                cell_number,
            )

        except Exception as exc:

            logger.exception(
                "Notebook cell %s failed",
                cell_number,
            )

            raise RuntimeError(
                f"Notebook cell {cell_number} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # Verify required runtime objects.
    required_objects = [
        "client",
        "COLLECTION_NAME",
        "dense_model",
        "sparse_model",
        "hybrid_search",
        "create_page_chunks",
        "prepare_qdrant_points",
        "index_chunks",
        "rag_agent",
        "TOP_K",
        "BATCH_SIZE",
    ]

    missing = [
        name
        for name in required_objects
        if name not in namespace
    ]

    if missing:
        raise RuntimeError(
            "Missing notebook runtime objects: "
            + ", ".join(missing)
        )

    # Validate Qdrant connectivity.
    client = namespace["client"]

    collection_name = namespace["COLLECTION_NAME"]

    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Qdrant collection '{collection_name}' does not exist."
        )

    collection_info = client.get_collection(
        collection_name
    )

    logger.info(
        "RAG runtime initialized successfully. Collection: %s",
        collection_name,
    )

    return {
        "namespace": namespace,
        "module": runtime_module,
        **{
            name: namespace[name]
            for name in required_objects
        },
        "collection_info": collection_info,
    }


def initialize_runtime():

    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Missing notebook: {NOTEBOOK_PATH}"
        )

    if not ENV_PATH.exists():
        raise FileNotFoundError(
            f"Missing configuration: {ENV_PATH}"
        )

    return load_agent_runtime(
        str(NOTEBOOK_PATH),
        str(PROJECT_DIR),
        NOTEBOOK_PATH.stat().st_mtime_ns,
        ENV_PATH.stat().st_mtime_ns,
    )


# ============================================================
# 4. PDF EXTRACTION
# ============================================================

def extract_pdf_pages(
    pdf_bytes: bytes,
    filename: str,
) -> list[dict]:

    if not pdf_bytes:
        raise ValueError(
            f"{filename}: uploaded file is empty."
        )

    file_size_mb = (
        len(pdf_bytes) / (1024 * 1024)
    )

    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"{filename}: exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )

    try:

        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as pdf:

            if pdf.is_encrypted:
                raise ValueError(
                    f"{filename}: encrypted PDFs are not supported."
                )

            if len(pdf) == 0:
                raise ValueError(
                    f"{filename}: PDF contains no pages."
                )

            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError(
                    f"{filename}: exceeds the "
                    f"{MAX_PDF_PAGES}-page limit."
                )

            pages = []

            for page_number, page in enumerate(
                pdf,
                start=1,
            ):

                page_text = page.get_text("text").strip()

                pages.append(
                    {
                        "page_no": page_number,
                        "text": page_text,
                    }
                )

    except (
        pymupdf.FileDataError,
        pymupdf.EmptyFileError,
    ) as exc:

        raise ValueError(
            f"{filename}: invalid or unreadable PDF."
        ) from exc

    if not any(
        page["text"]
        for page in pages
    ):
        raise ValueError(
            f"{filename}: no extractable text found. "
            "OCR may be required for this scanned PDF."
        )

    return pages


# ============================================================
# 5. DOCUMENT INGESTION
# ============================================================

def ingest_pdf(
    runtime: dict[str, Any],
    filename: str,
    pdf_bytes: bytes,
) -> dict[str, Any]:

    pages = extract_pdf_pages(
        pdf_bytes,
        filename,
    )

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(filename).stem,
    ).strip("._") or "document"

    content_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    document_title = (
        f"{safe_stem}-{content_hash[:12]}"
    )

    client = runtime["client"]

    collection_name = runtime["COLLECTION_NAME"]

    # Check whether this exact document was indexed earlier.
    existing_points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {
                    "key": "document_title",
                    "match": {
                        "value": document_title
                    },
                }
            ]
        },
        limit=1,
        with_payload=["document_title"],
        with_vectors=False,
    )

    if existing_points:

        return {
            "filename": filename,
            "document_title": document_title,
            "pages": len(pages),
            "chunks": 0,
            "status": "already_indexed",
        }

    # Use the notebook's existing chunking pipeline.
    chunks = runtime["create_page_chunks"](
        pages=pages,
        document_title=document_title,
        section_title="",
    )

    if not chunks:
        raise ValueError(
            f"{filename}: no chunks were generated."
        )

    # Generate stable Qdrant point IDs.
    points = runtime["prepare_qdrant_points"](
        chunks
    )

    if not points:
        raise ValueError(
            f"{filename}: no Qdrant points were generated."
        )

    # Use the notebook's existing dense/sparse indexing pipeline.
    runtime["index_chunks"](
        points,
        batch_size=runtime["BATCH_SIZE"],
    )

    return {
        "filename": filename,
        "document_title": document_title,
        "pages": len(pages),
        "chunks": len(points),
        "status": "indexed",
    }


# ============================================================
# 6. SESSION STATE
# ============================================================

def initialize_session():

    defaults = {
        "messages": [],
        "thread_id": str(uuid.uuid4()),
        "ingestion_results": [],
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.title("Document Research Agent")

st.caption(
    "LangGraph · Hybrid Retrieval · Qdrant"
)


# ============================================================
# 8. INITIALIZE RAG
# ============================================================

# Load the same environment file used by this Streamlit application.
# The notebook runtime may also load this file during its initialization.
load_dotenv(ENV_PATH, override=True)

try:

    runtime = initialize_runtime()

    # Initialize Langfuse after loading the environment and notebook runtime.
    # This ensures the SDK reads the configured public/secret keys and base URL.
    langfuse = get_client()
    langfuse_handler = CallbackHandler()

    logger.info("Langfuse client and LangChain callback initialized.")

except Exception as exc:

    logger.exception(
        "RAG runtime initialization failed"
    )

    st.error(
        f"RAG runtime initialization failed: {exc}"
    )

    st.info(
        "Check the terminal traceback for the exact notebook "
        "cell and exception."
    )

    st.stop()


client = runtime["client"]

collection_name = runtime["COLLECTION_NAME"]


# ============================================================
# 9. QDRANT STATUS
# ============================================================

try:

    collection_info = client.get_collection(
        collection_name
    )

    indexed_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

except Exception as exc:

    logger.exception(
        "Qdrant collection inspection failed."
    )

    st.error(
        f"Qdrant connection or collection error: {exc}"
    )

    st.stop()


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.subheader("Knowledge Base")

    st.metric(
        "Indexed chunks",
        f"{indexed_count:,}",
    )

    st.caption(
        f"Collection: `{collection_name}`"
    )

    st.caption(
        f"Status: `{collection_info.status}`"
    )

    st.divider()

    st.subheader("Document ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploads",
        help=(
            f"Maximum {MAX_FILE_SIZE_MB} MB per file. "
            f"Maximum {MAX_TOTAL_UPLOAD_MB} MB total."
        ),
    )

    if uploaded_files:

        total_bytes = sum(
            file.size
            for file in uploaded_files
        )

        total_size_mb = (
            total_bytes / (1024 * 1024)
        )

        st.caption(
            f"{len(uploaded_files)} files · "
            f"{total_size_mb:.1f} MB"
        )

        upload_limit_exceeded = (
            total_size_mb > MAX_TOTAL_UPLOAD_MB
        )

        if upload_limit_exceeded:

            st.error(
                f"Combined upload exceeds "
                f"{MAX_TOTAL_UPLOAD_MB} MB."
            )

        if st.button(
            "Index documents",
            type="primary",
            use_container_width=True,
            disabled=upload_limit_exceeded,
        ):

            results = []
            failures = []

            progress = st.progress(
                0.0,
                text="Starting ingestion...",
            )

            status_area = st.empty()

            for index, uploaded_file in enumerate(
                uploaded_files
            ):

                status_area.write(
                    f"Processing {uploaded_file.name} "
                    f"({index + 1}/{len(uploaded_files)})"
                )

                try:

                    result = ingest_pdf(
                        runtime=runtime,
                        filename=uploaded_file.name,
                        pdf_bytes=uploaded_file.getvalue(),
                    )

                    results.append(result)

                    logger.info(
                        "Document %s: %s",
                        uploaded_file.name,
                        result["status"],
                    )

                except Exception as exc:

                    logger.exception(
                        "Ingestion failed: %s",
                        uploaded_file.name,
                    )

                    failures.append(
                        (
                            uploaded_file.name,
                            str(exc),
                        )
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=(
                        f"Processed {index + 1}/"
                        f"{len(uploaded_files)}"
                    ),
                )

            st.session_state.ingestion_results = results

            if results:

                total_new_chunks = sum(
                    item["chunks"]
                    for item in results
                )

                st.success(
                    f"Processed {len(results)} document(s). "
                    f"New chunks indexed: {total_new_chunks:,}."
                )

            for filename, error in failures:

                st.error(
                    f"{filename}: {error}"
                )

            st.rerun()

    if st.session_state.ingestion_results:

        with st.expander("Latest ingestion results"):

            for result in st.session_state.ingestion_results:

                st.write(
                    f"**{result['filename']}** — "
                    f"{result['status']} · "
                    f"{result['pages']} pages · "
                    f"{result['chunks']} chunks"
                )

    st.divider()

    st.subheader("Conversation")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}`"
    )

    if st.button(
        "Start new conversation",
        use_container_width=True,
    ):

        st.session_state.thread_id = str(uuid.uuid4())

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "Clear displayed chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 11. CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# 12. LANGGRAPH AGENT INVOCATION
# ============================================================

user_input = st.chat_input(
    "Ask a question about your indexed documents..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Researching..."
            ):

                result = runtime["rag_agent"].invoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=user_input
                            )
                        ]
                    },
                    config={
                        "configurable": {
                            "thread_id": (
                                st.session_state.thread_id
                            )
                        },
                        "recursion_limit": 20,
                        "callbacks": [langfuse_handler],
                    },
                )

                # Ensure buffered Langfuse events are sent after this request.
                langfuse.flush()

                final_message = result["messages"][-1]

                response = getattr(
                    final_message,
                    "content",
                    str(final_message),
                )

                if not isinstance(response, str):
                    response = str(response)

                st.markdown(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

        except Exception:

            logger.exception(
                "LangGraph invocation failed."
            )

            st.error(
                "Agent execution failed. "
                "Check the terminal logs for details."
            )
