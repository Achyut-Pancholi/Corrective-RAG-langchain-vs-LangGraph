from dataclasses import dataclass

@dataclass
class TraceStep:
    """Represents a step in the execution trace."""
    name: str
    status: str  # 'success', 'warning', 'error'
    details: str | None = None

def format_docs(docs):
    """Formats a list of documents into a single string for prompts."""
    return "\n\n".join(doc.page_content for doc in docs)
