from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser

from src.models.state import GraphState
from src.retrieval.retriever import get_retriever
from src.prompts.templates import RAG_PROMPT, GRADING_PROMPT, REWRITE_PROMPT
from src.llm.factory import get_llm
from src.utils.formatting import format_docs, TraceStep
from src.config import MAX_RETRIES

def retrieve_node(state: GraphState):
    """Retrieve documents based on the current query."""
    question = state["rewritten_query"] if state.get("rewritten_query") else state["question"]
    trace = state.get("execution_trace", [])
    
    if not trace:
        trace.append(TraceStep(name="Question received", status="success"))
    
    retriever = get_retriever()
    documents = retriever.invoke(question)
    
    step_name = "Documents retrieved" if not state.get("rewritten_query") else "Documents retrieved again"
    trace.append(TraceStep(name=step_name, status="success", details=f"Retrieved {len(documents)} documents"))
    
    return {"documents": documents, "execution_trace": trace}

def grade_documents_node(state: GraphState):
    """Grade the retrieved documents using the LLM."""
    question = state["question"]
    documents = state["documents"]
    trace = state.get("execution_trace", [])
    
    llm = get_llm()
    chain = GRADING_PROMPT | llm | StrOutputParser()
    
    relevant_docs = []
    for doc in documents:
        # Prompt expects binary 'yes' or 'no'
        res = chain.invoke({"question": question, "document": doc.page_content})
        if "yes" in res.lower():
            relevant_docs.append(doc)
            
    if relevant_docs:
        trace.append(TraceStep(name="Documents graded", status="success", details="Sufficient context found"))
        return {"documents": relevant_docs, "execution_trace": trace}
    else:
        trace.append(TraceStep(name="Documents graded", status="warning", details="Documents insufficient"))
        return {"documents": [], "execution_trace": trace}

def rewrite_query_node(state: GraphState):
    """Rewrite the user's query when documents are insufficient."""
    question = state["question"]
    trace = state.get("execution_trace", [])
    retry_count = state.get("retry_count", 0) + 1
    
    llm = get_llm()
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    
    rewritten_query = chain.invoke({"question": question})
    
    trace.append(TraceStep(name="Query rewritten", status="warning", details=f"New query: {rewritten_query}"))
    
    return {"rewritten_query": rewritten_query, "retry_count": retry_count, "execution_trace": trace}

def generate_node(state: GraphState):
    """Generate the final answer using retrieved context."""
    question = state["question"]
    documents = state["documents"]
    trace = state.get("execution_trace", [])
    
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    
    context = format_docs(documents) if documents else "No relevant context found."
    generation = chain.invoke({"context": context, "question": question})
    
    trace.append(TraceStep(name="Answer generated", status="success"))
    
    return {"generation": generation, "execution_trace": trace}

def conditional_routing(state: GraphState):
    """Route to generate or rewrite based on grading and retry count."""
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)
    
    if documents:
        # We have relevant documents
        return "generate"
    
    if retry_count >= MAX_RETRIES:
        # Reached limit, force generation
        return "generate"
        
    # Insufficient documents and haven't hit limit
    return "rewrite"

def build_crag_graph():
    """Builds and compiles the CRAG StateGraph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        conditional_routing,
        {
            "generate": "generate",
            "rewrite": "rewrite_query"
        }
    )
    
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def run_langgraph_flow(question: str) -> GraphState:
    """Runs the CRAG flow via LangGraph."""
    app = build_crag_graph()
    
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_query": "",
        "execution_trace": []
    }
    
    # LangGraph returns a dict of the final state
    final_state = app.invoke(initial_state)
    return final_state
