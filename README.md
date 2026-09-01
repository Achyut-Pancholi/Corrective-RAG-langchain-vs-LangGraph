# Adaptive RAG (CRAG) POC

## Project Overview
This project is a Proof of Concept (POC) demonstrating **Corrective Retrieval-Augmented Generation (CRAG)**. It compares a standard linear RAG pipeline (using LangChain) against an adaptive, self-correcting RAG pipeline (using LangGraph).

## Problem
Standard RAG retrieves documents based on a user query and immediately generates an answer. If the retrieved documents are irrelevant or poor in quality, the system lacks a mechanism to correct itself, often leading to hallucinations or "I don't know" responses.

CRAG solves this by introducing a grading step. An LLM evaluates the retrieved documents for relevance. If they are deemed insufficient, the system dynamically rewrites the query and retrieves new documents before generating the final answer.

## Architecture
- **LangChain Flow (Baseline):** `Question → Retrieve → Generate`
- **LangGraph Flow (CRAG):** `Question → Retrieve → Grade → Conditional Routing (Rewrite → Retrieve if needed) → Generate`

Shared components include the LLM (Groq API), Embeddings (HuggingFace local), Vector Store (Chroma), and Prompts.

## Project Structure
- `data/documents/`: Small set of synthetic documents used for the knowledge base.
- `src/config.py`: Environment and constant configuration.
- `src/llm/factory.py`: LLM and Embedding initialization.
- `src/ingestion/loader.py`: Document loading and text splitting.
- `src/retrieval/`: Vector store and retriever setup.
- `src/prompts/`: System prompts for generation, grading, and rewriting.
- `src/models/state.py`: TypedDict for LangGraph state management.
- `src/flows/`: The core LangChain and LangGraph implementations.
- `app.py`: Streamlit User Interface.

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your Groq API key:
   ```env
   GROQ_API_KEY=your_key_here
   ```

## Run
Start the Streamlit application:
```bash
streamlit run app.py
```

## Example Questions to Try
Our knowledge base contains synthetic data about the "lost city of Aethelgard" and "Lumicite crystals".

1. **Normal Retrieval (Both succeed):**
   * "Who discovered the lost city of Aethelgard and when?"
2. **Corrective Retrieval (LangGraph corrects):**
   * "What did Marcus Thorne think about the new treaty regarding the blue crystals?" (Uses vague terms, triggering poor initial retrieval, prompting a rewrite).
3. **Unsupported Information (System declines):**
   * "What is the population of Aethelgard?"

## Limitations
- This POC uses a very small, fixed document set for demonstration purposes.
- Evaluation (grading) is qualitative, handled by the LLM.

## Future Improvements
- Automated quantitative evaluation (RAGAS).
- Integration with a production vector database (e.g., Pinecone, Qdrant).
- Hybrid search (Keyword + Semantic).
