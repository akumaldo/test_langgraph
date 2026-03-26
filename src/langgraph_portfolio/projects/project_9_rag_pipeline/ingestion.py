"""
PROJECT 9 — RAG Pipeline (LlamaIndex Workflows)
================================================
FILE: ingestion.py — IngestionWorkflow

CONCEPT: DOCUMENT INGESTION PIPELINE
Raw documents (PDFs) are too large to pass to an LLM. They need to be:
1. LOADED  — read the PDF, extract text page by page
2. CHUNKED — split into smaller pieces (paragraphs, sentences)
3. EMBEDDED — convert each chunk to a vector (numerical representation)
4. INDEXED — store vectors in a searchable structure

Compare to P2's RAG: we had a list of Document objects with keyword search.
That was "toy RAG." Real RAG uses embeddings and vector similarity — that's
what LlamaIndex provides out of the box.

CONCEPT: WORKFLOW COMPOSITION
This IngestionWorkflow runs ONCE at startup. Its output (the index + nodes)
feeds into the QueryWorkflow, which runs per question. This is workflow
composition — building bigger systems from smaller, focused workflows.
Like functions that do one thing well.

THE FLOW:
StartEvent(pdf_path) → load_document → DocumentLoaded → chunk_and_embed → StopEvent(index, nodes)
"""

from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PyMuPDFReader

from .events import DocumentLoaded

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# These constants control the ingestion pipeline. Tuning them affects
# retrieval quality:
# - CHUNK_SIZE: how many tokens per chunk. Too small = fragments lose context.
#   Too large = chunks contain irrelevant info alongside relevant info.
# - CHUNK_OVERLAP: tokens shared between consecutive chunks. Prevents
#   cutting a sentence in half — the overlap means both chunks have it.

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
LLM_MODEL = "qwen3.5:2b"
EMBEDDING_MODEL = "qwen3-embedding"


class IngestionWorkflow(Workflow):
    """Loads a PDF, chunks it, embeds the chunks, and builds a vector index.

    CONCEPT: @step DECORATOR
    Each method decorated with @step becomes a workflow step. The type hint
    on the parameter tells LlamaIndex which event triggers this step:

        @step
        async def my_step(self, ev: SomeEvent) -> AnotherEvent:

    This means: "when SomeEvent is emitted, run my_step and it will produce
    AnotherEvent." The workflow engine handles the routing — you never call
    steps directly.

    Compare to LangGraph where you'd write:
        graph.add_node("my_step", my_step_fn)
        graph.add_edge("previous_step", "my_step")

    Here, the connection is IMPLICIT through event types.
    """

    @step
    async def load_document(self, ev: StartEvent) -> DocumentLoaded:
        """Step 1: Load the PDF file and extract text.

        CONCEPT: StartEvent
        Every workflow begins with StartEvent. You pass data to it when
        calling workflow.run(). The fields you set on StartEvent are
        accessible as attributes (ev.pdf_path).

        PyMuPDFReader extracts text page by page, preserving page numbers
        in metadata. This is important later — when we show retrieved chunks,
        we can tell the user which page they came from.
        """
        pdf_path = ev.get("pdf_path")
        if not pdf_path:
            raise ValueError("pdf_path is required in StartEvent")

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        print(f"  Loading PDF: {path.name}")

        reader = PyMuPDFReader()
        raw_docs = reader.load_data(str(path))

        if not raw_docs:
            raise ValueError(f"No text extracted from {path.name}. Is the PDF empty or image-only?")

        print(f"  Extracted {len(raw_docs)} pages")

        # Emit DocumentLoaded — the chunk_and_embed step is listening for this
        return DocumentLoaded(raw_docs=raw_docs)

    @step
    async def chunk_and_embed(self, ev: DocumentLoaded) -> StopEvent:
        """Step 2: Chunk documents, embed them, and build the vector index.

        CONCEPT: CHUNKING
        A 50-page PDF might have 50,000 tokens. An LLM can't process all
        of that at once, and most of it isn't relevant to a specific question.
        Chunking splits the document into smaller pieces (e.g., 512 tokens each).

        SentenceSplitter is smart about boundaries — it tries to break at
        sentence endings, not mid-word. The overlap means consecutive chunks
        share some text, so information at chunk boundaries isn't lost.

        CONCEPT: EMBEDDINGS
        Each chunk gets converted to a VECTOR — a list of numbers (e.g., 768
        dimensions for qwen3-embedding). Chunks with similar meaning have
        similar vectors. When you search, your query also becomes a vector,
        and we find chunks whose vectors are closest to the query's vector.

        This is fundamentally different from keyword search (P2):
        - Keyword: "machine learning" only finds chunks containing those exact words
        - Semantic: "ML algorithms" finds chunks about machine learning too,
          because the MEANING is similar even if the words differ

        CONCEPT: VectorStoreIndex
        LlamaIndex's VectorStoreIndex handles the entire embed+store pipeline.
        Under the hood it: (1) calls the embedding model for each chunk,
        (2) stores the vectors in memory, (3) provides a retriever interface.
        """
        print("  Chunking documents...")

        # Configure the global settings for LlamaIndex
        # These settings are used by VectorStoreIndex when building the index
        Settings.llm = Ollama(model=LLM_MODEL, request_timeout=600)
        Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

        # Split documents into chunks
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(ev.raw_docs)

        if not nodes:
            raise ValueError("Chunking produced 0 chunks. The PDF might have no extractable text.")

        print(f"  Created {len(nodes)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        print("  Embedding chunks (this may take a moment)...")

        # Build the vector index — this triggers embedding for every chunk
        # The index stores both the vectors AND the original text
        index = VectorStoreIndex(nodes=nodes)

        print(f"  Index ready ({len(nodes)} chunks embedded)")

        # StopEvent carries the result back to the caller (workflow.run())
        # We return both the index (for vector search) and the raw nodes
        # (needed by the BM25 keyword retriever, which works on text, not vectors)
        return StopEvent(result={"index": index, "nodes": nodes})
