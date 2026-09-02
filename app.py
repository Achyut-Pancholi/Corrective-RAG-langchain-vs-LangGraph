import streamlit as st
import os

# Load env vars before importing flows (since factory.py checks it)
from dotenv import load_dotenv
load_dotenv()

from src.flows.langchain_rag import run_langchain_flow
from src.flows.langgraph_crag import run_langgraph_flow
from src.utils.formatting import TraceStep

st.set_page_config(page_title="Adaptive RAG - CRAG POC", layout="centered")

st.title("Adaptive RAG — CRAG POC")
st.markdown("Compare standard linear RAG against self-correcting graph RAG.")

# API Key Check
if not os.getenv("GROQ_API_KEY"):
    st.error("Missing GROQ_API_KEY in environment variables. Please check your .env file or add it here.")
    st.stop()

# with st.sidebar:
#     st.header("Architecture Visualizer")
#     st.subheader("LangChain Baseline")
#     st.components.v1.html(
#         """
#         <script type="module">
#           import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
#           mermaid.initialize({ startOnLoad: true, theme: 'dark' });
#         </script>
#         <div class="mermaid">
#         graph LR;
#             A([Start]) --> B[1. Retrieve] --> C[2. Generate] --> D([End]);
#         </div>
#         """,
#         height=140,
#     )

#     st.divider()
#     st.subheader("LangGraph CRAG Loop")
#     st.components.v1.html(
#         """
#         <script type="module">
#           import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
#           mermaid.initialize({ startOnLoad: true, theme: 'dark' });
#         </script>
#         <div class="mermaid">
#         graph TD;
#             Start([Start]) --> Retrieve[1. Retrieve Docs];
#             Retrieve --> Grade{2. Grade Relevance};
#             Grade -- Relevant --> Generate[4. Generate Answer];
#             Grade -- Insufficient --> Rewrite[3. Rewrite Query];
#             Rewrite --> Retrieve;
#             Generate --> End([End]);
#         </div>
#         """,
#         height=320,
#     )
#     st.divider()

flow_choice = st.radio(
    "Select Flow",
    ["LangChain — Baseline RAG", "LangGraph — Corrective RAG"],
    horizontal=True
)

question = st.text_input("Question", placeholder="Ask a question about Aethelgard or Lumicite crystals...")

if st.button("Ask Question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            try:
                if "LangChain" in flow_choice:
                    result = run_langchain_flow(question)
                else:
                    result = run_langgraph_flow(question)
                
                st.divider()
                st.subheader("Answer")
                st.write(result["generation"])
                
                st.divider()
                st.subheader("Sources")
                if result.get("documents"):
                    for i, doc in enumerate(result["documents"], 1):
                        with st.expander(f"Source {i}"):
                            st.write(doc.page_content)
                else:
                    st.write("No relevant sources found.")
                
                st.divider()
                st.subheader("Execution Trace")
                for step in result.get("execution_trace", []):
                    icon = "✅" if step.status == "success" else "⚠️" if step.status == "warning" else "❌"
                    details_text = f" - *{step.details}*" if step.details else ""
                    st.markdown(f"{icon} **{step.name}**{details_text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
