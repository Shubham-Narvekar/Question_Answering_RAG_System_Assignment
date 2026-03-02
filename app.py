import streamlit as st
from processing.rag_pipeline import SwiggyRAG
from processing.retriever import SwiggyRetriever


st.set_page_config(page_title="Swiggy Annual Report RAG", layout="wide")

st.title("📊 Swiggy Annual Report Q&A System")
st.markdown("Ask questions based strictly on the Swiggy Annual Report (FY 2023–24).")

# Initialize RAG once
@st.cache_resource
def load_rag():
    return SwiggyRAG()

rag = load_rag()

# Input field
question = st.text_input("Enter your question:")

if st.button("Ask") and question:
    with st.spinner("Retrieving answer..."):
        answer = rag.answer(question)

        # Also retrieve chunks for display
        retriever = SwiggyRetriever(
            "output/faiss_index/swiggy_faiss.index",
            "output/faiss_index/metadata.json"
        )
        retrieved_chunks = retriever.search(question, top_k=3, similarity_threshold=0.40)

    st.subheader("📌 Answer")
    st.write(answer)

    if retrieved_chunks:
        st.subheader("🔍 Supporting Context")
        for i, chunk in enumerate(retrieved_chunks):
            with st.expander(f"Source {i+1} | Page {chunk['page_number']} | Similarity: {chunk['similarity_score']:.4f}"):
                st.write(chunk["chunk_text"])