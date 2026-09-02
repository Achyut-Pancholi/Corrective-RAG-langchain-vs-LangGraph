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
    """You are a strict relevance grader assessing whether a retrieved document contains specific facts or context to answer the user's question.
If the document contains factual information, data, or direct evidence that helps answer the question, grade it as 'yes'.
If the document only mentions tangential keywords without providing substantive facts to answer the question, grade it as 'no'.

Provide the output as exactly 'yes' or 'no' with no other text.

Question: {question}
Document: {document}
Relevant (yes/no):"""
)

# Query Rewriting Prompt (Domain-Agnostic / Industry Standard)
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert search query rewriter for an advanced information retrieval system.
A user asked a question, but initial retrieval returned irrelevant documents.
Analyze the user's question to identify the core technical concepts, decompose vague phrasing, and formulate a targeted,
high-relevance search query.
Focus on probable technical terminology, alternative keywords, and underlying mechanisms.

Initial question: {question}

Provide only the reformulated plain-text search query. Do NOT use markdown, quotes, explanations, or boolean logic:"""
)
