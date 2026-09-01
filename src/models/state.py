from typing import TypedDict, List
from langchain_core.documents import Document
from src.utils.formatting import TraceStep

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: The user's question
        documents: List of retrieved documents
        generation: The final generated answer
        retry_count: Number of times retrieval has been retried
        rewritten_query: The rewritten question (if applicable)
        execution_trace: List of TraceStep objects detailing execution
    """
    question: str
    documents: List[Document]
    generation: str
    retry_count: int
    rewritten_query: str
    execution_trace: List[TraceStep]
