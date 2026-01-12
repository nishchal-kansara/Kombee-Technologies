import sys
import os
import uuid
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ingest_upload import ingest_uploaded_file
from backend.rag import load_qa

st.title("Private Company ChatGPT")

doc_option = st.radio(
    "Select document source",
    ["Use company preloaded documents", "Upload your own document"]
)

index_path = None

if doc_option == "Use company preloaded documents":
    index_path = "faiss_preload_index"
else:
    uploaded_file = st.file_uploader("Upload a txt file", type=["txt"])
    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}.txt"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        index_path = ingest_uploaded_file(file_path, f"faiss_upload_{file_id}")

question = st.text_input("Ask a question")

if st.button("Ask"):
    if question and index_path:
        qa = load_qa(index_path)
        answer = qa.invoke(question).content
        st.write(answer)
