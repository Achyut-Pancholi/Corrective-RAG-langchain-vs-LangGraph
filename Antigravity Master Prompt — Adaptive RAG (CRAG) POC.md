# Master Prompt: Build Adaptive RAG (CRAG) POC — LangChain vs LangGraph

You are a senior Python/Generative AI engineer working on a small internal POC.

Build a **clean, maintainable, human-written Python application** for the following problem:

## 1. Problem Statement

Standard RAG retrieves documents once and immediately generates an answer. If the retrieved documents are poor or irrelevant, there is no correction mechanism.

This POC should demonstrate **Corrective RAG (CRAG)**, where the system:

1. Receives a user question.
2. Retrieves relevant documents.
3. Uses an LLM to judge whether the retrieved documents are sufficient/relevant.
4. If sufficient, generates the final answer.
5. If insufficient, rewrites the query.
6. Retrieves again using the rewritten query.
7. Generates the final answer.
8. Uses a capped retry count so the workflow always terminates.

The purpose is NOT to present LangChain and LangGraph as competing libraries.

The comparison is:

- **Flow A:** Linear RAG using LangChain.
- **Flow B:** Corrective/Adaptive RAG using LangGraph.

LangChain should provide the common RAG components in both flows, while LangGraph should be responsible for stateful orchestration, conditional branching, and the corrective loop.

The same document set, retriever, vector store, embeddings, LLM configuration, prompts where appropriate, and retrieval parameters should be shared between both flows so that the main difference is the orchestration logic.

---

# 2. Primary Goal

Build a working Streamlit application where the user can:

1. Select **LangChain** or **LangGraph**.
2. Enter a question.
3. Execute the selected flow.
4. See the final answer.
5. See retrieved source information.
6. See an understandable execution trace.

The UI should make the difference between the two approaches obvious.

The application should be simple enough to complete and understand within approximately **7 working days**.

Do NOT over-engineer the project.

---

# 3. Expected Architecture

Use a clean separation of concerns.

Recommended high-level structure:

```text
User
  |
  v
Streamlit UI
  |
  +-------------------------+
  |                         |
  v                         v
LangChain Flow          LangGraph Flow
  |                         |
  |                         v
  |                    Retrieve
  |                         |
  |                    Grade Documents
  |                         |
  |                  +------+------+
  |                  |             |
  |              Relevant       Not Relevant
  |                  |             |
  |                  v             v
  |              Generate      Rewrite Query
  |                                |
  |                                v
  |                             Retrieve
  |                                |
  |                                v
  |                             Generate
  |                                |
  +----------------+---------------+
                   |
                   v
              Final Answer
                   |
                   v
              Source Details
                   |
                   v
            Execution Trace
```

---

# 4. Flow A — LangChain Baseline

Implement a simple, deterministic RAG pipeline.

Expected behavior:

```text
Question
   ↓
Retriever
   ↓
Top-K Documents
   ↓
Prompt
   ↓
LLM
   ↓
Final Answer
```

Use LangChain/LangChain-supported components for:

- Document loading
- Text splitting
- Embeddings
- Vector store
- Retriever
- Prompt construction
- Chat model
- RAG generation

This flow should intentionally have **no corrective loop**.

If retrieval is poor, the baseline should simply proceed to generation.

This limitation is important because it establishes the reason for the CRAG flow.

---

# 5. Flow B — LangGraph CRAG

Implement the corrective workflow using LangGraph `StateGraph`.

Expected graph:

```text
START
  ↓
retrieve
  ↓
grade_documents
  ↓
conditional routing
  |
  +---- relevant ----> generate ----> END
  |
  +---- not relevant -> rewrite_query
                          |
                          v
                       retrieve
                          |
                          v
                     grade_documents
                          |
                          v
                       generate
                          |
                          v
                         END
```

Use a maximum of **2 retrieval attempts**.

Do not create an infinite loop.

The graph should maintain explicit state.

At minimum, state should contain information such as:

```text
question
documents
generation
retry_count
rewritten_query
execution_trace
```

Use an appropriate typed state representation, preferably `TypedDict` or another clean Python typing approach.

---

# 6. Grading Logic

The grading step should use the LLM to determine whether retrieved documents are relevant/sufficient for answering the user's question.

Prefer structured output rather than parsing arbitrary natural-language responses.

For example:

```text
relevant: true/false
reason: short explanation
```

The grading implementation must be robust.

Avoid fragile logic such as:

```python
if "yes" in response:
```

Prefer structured/schema-based output when supported by the selected model.

If structured output is not supported reliably by the chosen model, implement a small, well-isolated fallback parser.

---

# 7. Query Rewriting

When documents are judged insufficient:

```text
Original Question
        ↓
Rewrite Query
        ↓
Improved Search Query
        ↓
Retriever
```

The rewritten query should be optimized for document retrieval rather than answering the question directly.

Keep the rewriting prompt simple and deterministic.

Do not build a complex autonomous agent.

---

# 8. Shared Components

Both flows MUST share common infrastructure wherever possible.

Create reusable components for:

- Configuration
- LLM initialization
- Embedding initialization
- Document loading
- Text splitting
- Vector store initialization
- Retriever creation
- RAG prompt
- Document formatting
- Common response models/data structures

Do not duplicate the same setup code in `langchain_flow.py` and `langgraph_flow.py`.

The architecture should make it obvious that:

```text
                    Shared
        ┌─────────────┼─────────────┐
        │             │             │
    Documents      Retriever       LLM
        │             │             │
        └─────────────┼─────────────┘
                      │
              Orchestration differs
                  /           \
                 /             \
          LangChain          LangGraph
```

---

# 9. Suggested Project Structure

Use a structure similar to:

```text
project/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── .gitignore
│
├── data/
│   └── documents/
│
├── src/
│   ├── __init__.py
│   │
│   ├── config.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── state.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── templates.py
│   │
│   ├── flows/
│   │   ├── __init__.py
│   │   ├── langchain_rag.py
│   │   └── langgraph_crag.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── formatting.py
│
└── tests/
    ├── test_retrieval.py
    └── test_flows.py
```

You may adjust the structure if there is a genuinely better organization, but maintain the same separation of concerns.

Do not create unnecessary layers or abstractions just for the sake of having more files.

---

# 10. Python Coding Standards

Write code as an experienced human Python developer would.

Follow modern Python best practices:

- Clear naming
- Small focused functions
- Type hints
- Useful docstrings
- Sensible exception handling
- No unnecessary classes
- No unnecessary abstraction
- No global mutable state
- No duplicated logic
- No magic numbers
- Constants/configuration in appropriate locations
- Use `pathlib` for filesystem paths
- Use environment variables for secrets
- Keep UI logic separate from business logic
- Prefer dependency injection where it actually improves testability
- Keep functions reasonably small
- Follow PEP 8
- Use readable imports
- Avoid wildcard imports
- Avoid deeply nested logic

Use Python typing wherever it improves clarity.

Example:

```python
def retrieve_documents(query: str, k: int) -> list[Document]:
    ...
```

rather than untyped functions.

---

# 11. Human-Written Code Requirement

The code should look like it was written and maintained by a competent developer, not generated by an AI trying to explain every line.

IMPORTANT:

Do NOT add excessive comments.

Do NOT write comments like:

```python
# Import the os module
import os
```

Do NOT comment obvious code.

Use comments only when they explain:

- Why a non-obvious decision was made
- A limitation
- An important business/technical constraint
- Something that future maintainers genuinely need to know

Prefer good function/variable names and clean structure over comments.

Docstrings should be used selectively for public/reusable functions where they add value.

Do not put a giant explanatory comment above every function.

---

# 12. Configuration and Secrets

Never hardcode API keys.

Use `.env`.

Provide:

```text
.env.example
```

with placeholders such as:

```text
LLM_API_KEY=
LLM_MODEL=
EMBEDDING_MODEL=
```

Do not commit `.env`.

Add `.env` to `.gitignore`.

The application should fail with a clear, human-readable configuration error if required environment variables are missing.

Do not print secrets.

---

# 13. LLM Provider

Choose one practical LangChain-supported chat model that can be configured through environment variables.

Keep the LLM integration isolated so it can be changed easily later.

Do not implement multiple providers unless there is a strong reason.

The POC should prioritize:

1. Reliability
2. Simplicity
3. Easy local setup
4. Clear code

over provider flexibility.

---

# 14. Embeddings and Vector Store

Use either:

- Chroma
- FAISS

Choose whichever provides the simplest reliable local setup.

Use a LangChain-supported embedding model.

The vector store should be created from a small fixed document set.

Do not introduce a hosted vector database.

Do not introduce unnecessary infrastructure.

---

# 15. Documents

Use a small fixed document collection suitable for demonstrating RAG.

The documents should contain enough information to demonstrate:

### Case A — Both flows work

A question whose answer is clearly present in the retrieved documents.

Expected:

```text
LangChain → correct answer

LangGraph → correct answer
```

### Case B — LangGraph correction

A question where the first retrieval is intentionally weak or irrelevant enough that the grading step can trigger query rewriting.

Expected:

```text
LangChain
Question → Retrieve → Generate

LangGraph
Question → Retrieve → Grade
                    ↓
                 Not relevant
                    ↓
                 Rewrite
                    ↓
                 Retrieve
                    ↓
                 Grade/Generate
```

### Case C — Information unavailable

A question whose answer is not supported by the documents.

The system should avoid confidently inventing information.

The final answer should clearly indicate when the available documents do not contain sufficient information.

Do not pretend the RAG system has knowledge it did not retrieve from the supplied documents.

---

# 16. Retrieval Configuration

Keep retrieval parameters centralized.

For example:

```text
TOP_K = 4
MAX_RETRIES = 1
```

Remember that "capped at 2 attempts" means the system should make no more than two retrieval attempts in the corrective flow.

Make this behavior explicit in code.

Do not scatter values such as `4`, `2`, etc. throughout the project.

---

# 17. Streamlit UI

Keep the UI intentionally simple and professional.

Suggested layout:

```text
--------------------------------------------------
          Adaptive RAG — CRAG POC
--------------------------------------------------

Select Flow

( ) LangChain — Baseline RAG
( ) LangGraph — Corrective RAG

Question
[                                        ]

              [ Ask Question ]

--------------------------------------------------
Answer

...

--------------------------------------------------
Sources

1. Document / Page / Chunk
2. Document / Page / Chunk

--------------------------------------------------
Execution Trace

✓ Question received
✓ Documents retrieved
✓ Documents graded
✓ Query accepted
✓ Answer generated
--------------------------------------------------
```

For LangChain:

```text
Execution Trace

✓ Question received
✓ Documents retrieved
✓ Answer generated
```

For LangGraph:

```text
Execution Trace

✓ Question received
✓ Documents retrieved
✓ Documents graded
⚠ Documents insufficient
✓ Query rewritten
✓ Documents retrieved again
✓ Documents graded
✓ Answer generated
```

The execution trace should be generated by the application logic, not manually hardcoded in the UI.

---

# 18. UI Requirements

The UI should contain:

### Required

- Flow selector
- Question input
- Ask/Submit button
- Answer section
- Sources section
- Execution trace

### Optional

- Retrieved document previews
- Rewritten query
- Retrieval attempt number
- Grade/relevance reason
- Execution time

Only add optional elements if they improve the demo without significantly increasing complexity.

Do not turn the UI into a dashboard.

---

# 19. Execution Trace Design

Create a small reusable trace representation.

For example:

```python
@dataclass
class TraceStep:
    name: str
    status: str
    details: str | None = None
```

Or use another clean approach if better suited.

Possible statuses:

```text
success
warning
error
```

The trace should represent what actually happened.

Never show:

```text
✓ Query rewritten
```

if the query was not actually rewritten.

For LangGraph, trace important state transitions.

---

# 20. Error Handling

Handle expected failures gracefully.

Examples:

- Missing API key
- Missing documents
- Vector store initialization failure
- LLM failure
- Retrieval failure
- Invalid structured grading response
- Empty user question

The Streamlit UI should display a useful user-facing message.

Do not expose raw stack traces to normal users.

However, make debugging information available through logging.

---

# 21. Logging

Use Python's `logging` module.

Do not use `print()` throughout the application for application logging.

Use sensible logging levels:

```text
DEBUG
INFO
WARNING
ERROR
```

Do not log:

- API keys
- secrets
- unnecessary full document contents
- sensitive user information

Logging should help a developer troubleshoot the application.

---

# 22. Testing

Do not attempt to build a huge test suite.

Create focused tests for the most important logic.

At minimum test:

1. Retrieval initialization
2. LangChain flow returns a response
3. LangGraph flow returns a response
4. LangGraph can take the relevant branch
5. LangGraph can take the rewrite branch
6. Retry limit is respected
7. Empty question is handled

Where practical, mock LLM responses in unit tests rather than making API calls for every test.

Keep tests simple and understandable.

---

# 23. README

Create a concise but professional README containing:

## Project Overview

What the POC does.

## Problem

Why standard RAG can fail when retrieval is poor.

## Architecture

Explain:

```text
LangChain → linear RAG

LangGraph → stateful CRAG with conditional correction
```

## Project Structure

Explain the major directories.

## Setup

Include:

```text
python -m venv .venv
```

activation instructions for Windows and Unix-like systems.

Then:

```text
pip install -r requirements.txt
```

Explain `.env` configuration.

## Run

Provide the command to launch Streamlit.

## Example Questions

Include several questions that demonstrate:

- Normal retrieval
- Corrective retrieval
- Unsupported information

## LangChain vs LangGraph

Explain the architectural difference clearly.

## Limitations

Mention that this is a POC using a small fixed document set and qualitative/manual evaluation.

## Future Improvements

Potentially mention:

- Automated evaluation
- Better retrieval strategies
- Reranking
- Hybrid search
- Production vector database
- Observability
- Authentication

Do not implement those features.

---

# 24. Dependency Management

Create a clean `requirements.txt`.

Only include dependencies that are actually used.

Do not install libraries "just in case."

Before finalizing:

- Remove unused imports
- Remove unused dependencies
- Check for deprecated APIs
- Use currently supported APIs for the installed package versions

Pin versions only where there is a good reason, especially if reproducibility matters.

---

# 25. Important LangChain/LangGraph Principle

Keep this distinction clear in the implementation:

### LangChain

Used for:

```text
LLM
Prompts
Embeddings
Documents
Vector Store
Retriever
RAG Chain
```

### LangGraph

Used for:

```text
State
Nodes
Edges
Conditional Routing
Loop / Retry
Execution State
```

Do not unnecessarily rebuild LangChain functionality inside LangGraph.

The LangGraph flow should compose LangChain components.

---

# 26. Avoid Fake "Agentic" Complexity

This is a POC about corrective RAG.

Do NOT add:

- Multi-agent systems
- Tool calling
- Autonomous planning
- Memory
- Web browsing
- Multiple LLMs
- Complex agent executors
- Human-in-the-loop workflows
- Long-term memory
- Authentication
- Database-backed users
- Background workers

unless absolutely necessary for the core requirement.

The graph should remain understandable.

---

# 27. Quality Requirements

Before considering the implementation complete, verify:

### Code

- [ ] Code is readable without extensive comments.
- [ ] Functions have clear responsibilities.
- [ ] No unnecessary duplication.
- [ ] No hardcoded secrets.
- [ ] No unused imports.
- [ ] No unnecessary dependencies.
- [ ] Type hints are used appropriately.
- [ ] Error handling is sensible.
- [ ] Logging is used appropriately.

### LangChain Flow

- [ ] Retrieves documents.
- [ ] Passes retrieved context to LLM.
- [ ] Produces an answer.
- [ ] Returns source information.
- [ ] Does not perform corrective retrieval.

### LangGraph Flow

- [ ] Uses StateGraph.
- [ ] Has explicit state.
- [ ] Retrieves documents.
- [ ] Grades document relevance.
- [ ] Uses conditional routing.
- [ ] Rewrites query when retrieval is insufficient.
- [ ] Retrieves again.
- [ ] Has a strict retry limit.
- [ ] Produces final answer.
- [ ] Records actual execution trace.

### UI

- [ ] User can select LangChain.
- [ ] User can select LangGraph.
- [ ] User can enter a question.
- [ ] Selected flow actually executes.
- [ ] Answer is displayed.
- [ ] Sources are displayed.
- [ ] Execution trace is displayed.
- [ ] Errors are presented clearly.

---

# 28. Development Approach

Do not try to build everything at once.

Implement in this order:

### Phase 1

Project setup + configuration.

### Phase 2

Document ingestion + embeddings + vector store + retriever.

### Phase 3

Working LangChain RAG baseline.

### Phase 4

LangGraph state + retrieve node + generate node.

### Phase 5

Document grading + conditional routing.

### Phase 6

Query rewriting + retry limit.

### Phase 7

Integrate both flows into Streamlit.

### Phase 8

Testing, cleanup, README, and demo polish.

After each phase, verify that the application still works.

---

# 29. Do Not Overwrite Existing Work Blindly

Before making changes:

1. Inspect the existing project.
2. Identify what already exists.
3. Reuse working components when appropriate.
4. Do not delete existing code unless necessary.
5. Do not introduce a completely different architecture without reason.
6. Preserve useful existing configuration.

If the project is empty, create the architecture described above.

If existing code is present, adapt it carefully.

---

# 30. Final Validation

Before declaring the project complete, actually run it.

Perform an end-to-end test for:

### Test 1

```text
Flow: LangChain
Question: known question
Expected: answer + sources
```

### Test 2

```text
Flow: LangGraph
Question: known question
Expected:
retrieve → grade → generate
```

### Test 3

```text
Flow: LangGraph
Question: question causing weak retrieval
Expected:
retrieve → grade → rewrite → retrieve → generate
```

### Test 4

```text
Flow: LangGraph
Question: unsupported question
Expected:
system does not fabricate unsupported information
```

### Test 5

```text
Empty question
Expected:
clear validation message
```

Also verify that the retry limit prevents infinite loops.

---

# 31. Seven-Day Scope Constraint

This is a small POC.

Prioritize:

```text
Working > Fancy
Readable > Clever
Simple > Over-engineered
Reliable > Feature-heavy
Demonstrable > Production-ready
```

The final application should be something a developer can understand by opening the repository and reading the code.

Do not spend time building production infrastructure.

The goal is a **clean internal POC demonstrating the architectural difference between linear RAG and corrective graph-based RAG.**

---

# 32. Final Deliverables

The finished project should contain:

```text
✓ Working Python application
✓ LangChain baseline RAG
✓ LangGraph CRAG
✓ Shared retrieval/LLM components
✓ Small fixed document set
✓ Streamlit UI
✓ Flow selector
✓ Question input
✓ Answer output
✓ Source information
✓ Real execution trace
✓ Query rewriting
✓ Conditional routing
✓ Retry limit
✓ Basic tests
✓ README
✓ requirements.txt
✓ .env.example
✓ .gitignore
```

---

# 33. Final Instruction

Act as a senior engineer, but keep the implementation appropriate for a **7-day internal POC**.

Do not blindly follow every architectural suggestion above if it makes the project unnecessarily complicated. Prefer the simplest implementation that satisfies the requirements.

Before coding, inspect the repository and existing environment.

Then implement the project incrementally.

After implementation:

1. Run the application.
2. Run the tests.
3. Fix errors.
4. Review the project for unnecessary complexity.
5. Review the code for duplication and readability.
6. Verify both LangChain and LangGraph flows independently.
7. Verify the corrective loop actually executes.
8. Verify the retry limit.
9. Verify the Streamlit UI.
10. Update the README with the actual setup/run instructions.

Do not merely create files and assume they work.

**The final standard is: a clean, understandable, working POC that I can confidently explain to my mentor line-by-line.**