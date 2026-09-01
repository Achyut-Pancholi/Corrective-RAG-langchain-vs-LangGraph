# Architecture, Decisions, and Context

This document maintains a living history of the architecture, key decisions, and context for the CRAG POC project. It is intended to be updated continuously as the project evolves.

## 1. Context

Standard Retrieval-Augmented Generation (RAG) processes retrieve documents once and immediately generate an answer. If retrieval yields irrelevant or insufficient documents, the final generation will either hallucinate or fail gracefully, but it cannot correct itself.

This POC demonstrates **Corrective RAG (CRAG)**, utilizing a graph-based state machine (LangGraph) to evaluate retrieved documents. If they are insufficient, the query is rewritten, and a secondary retrieval is performed before generation.

## 2. Architecture

The project splits into two flows that share a common infrastructure to highlight the difference in orchestration:

- **Flow A (LangChain Baseline):** `Question -> Retriever -> LLM -> Answer`
- **Flow B (LangGraph CRAG):** `Question -> Retriever -> Grade -> (Conditional: Rewrite & Retrieve again if needed) -> LLM -> Answer`

### Shared Infrastructure
- **Embeddings:** `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` via `langchain-huggingface` and `sentence-transformers` — runs locally, converting text into 384-dimensional dense vector embeddings.
- **Vector Store:** Chroma (`chromadb` & `langchain-chroma`) — runs locally in-memory, providing real-time vector search.
- **LLM Provider:** Groq (`langchain-groq`) — fast, free-tier accessible, using `openai/gpt-oss-20b` for reasoning, document grading, query rewriting, and final generation.

### App Structure
All code is organized under `src/` (`src/flows`, `src/ingestion`, `src/llm`, `src/retrieval`, `src/prompts`, `src/utils`), with `app.py` at the project root.

## 3. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-27 | Use Groq for LLM and HuggingFace for Embeddings | The user requested free options. Groq is fast and has generous free limits. HuggingFace provides local embeddings via `sentence-transformers`, avoiding costs. |
| 2026-08-27 | Direct folder structure | User explicitly requested clear structure to keep things simple. |
| 2026-08-27 | Synthetic Documents | A small synthetic document set guarantees we can reliably demonstrate the specific test cases. |
| 2026-08-29 | Add `langchain-chroma` & `sentence-transformers` | Added explicitly to `requirements.txt` to support modern LangChain Chroma vector store integration and Hugging Face local embeddings (`all-MiniLM-L6-v2`). |
| 2026-08-29 | Add `testing.txt` & `doc4.txt` | Documented interactive test cases, example Q&A, and added `doc4.txt` (Project Nexus) to test multi-document ingestion. |
