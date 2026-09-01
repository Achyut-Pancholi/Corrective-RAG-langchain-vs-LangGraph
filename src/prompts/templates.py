from langchain_core.prompts import ChatPromptTemplate

# RAG Generation Prompt
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If the answer is not contained in the context, say that you don't know based on the provided documents. Do not invent information.

Question: {question} 

Context:
{context}

Answer:"""
)

# Document Grading Prompt
GRADING_PROMPT = ChatPromptTemplate.from_template(
    """You are a grader assessing whether a retrieved document contains relevant information to answer a user's question.
If the document contains facts or information that help answer the question, grade it as 'yes'. Otherwise, grade it as 'no'.
Provide the output as exactly 'yes' or 'no' with no other text.

Question: {question}
Document: {document}
Relevant (yes/no):"""
)

# Query Rewriting Prompt
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a question re-writer for a vector database containing documents about Lumicite, Zephyr Corporation, and Project Nexus.
Convert the user's question into a plain-text search query using terms like 'Zephyr Corporation consumer medical devices' or 'Lumicite Treaty'.
Do NOT use markdown, code blocks, or boolean logic like AND/OR. Output ONLY plain text.

Initial question: {question}
Search Query:"""
)
