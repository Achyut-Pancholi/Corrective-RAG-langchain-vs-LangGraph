from langchain_core.output_parsers import StrOutputParser
from src.retrieval.retriever import get_retriever
from src.prompts.templates import RAG_PROMPT
from src.llm.factory import get_llm
from src.utils.formatting import format_docs, TraceStep
from src.models.state import GraphState

def run_langchain_flow(question: str) -> GraphState:
    """Runs the linear baseline RAG flow using LangChain."""
    
    trace = [TraceStep(name="Question received", status="success")]
    
    retriever = get_retriever()
    llm = get_llm()
    
    # 1. Retrieve
    docs = retriever.invoke(question)
    trace.append(TraceStep(name="Documents retrieved", status="success", details=f"Retrieved {len(docs)} documents"))
    
    # 2. Generate (Sequential LCEL chain)
    chain = RAG_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({
        "context": format_docs(docs),
        "question": question
    })
    trace.append(TraceStep(name="Answer generated", status="success"))
    
    return {
        "question": question,
        "documents": docs,
        "generation": generation,
        "retry_count": 0,
        "rewritten_query": "",
        "execution_trace": trace
    }
