import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ingest_preloaded_docs():
    data_path = os.path.join(PROJECT_ROOT, "data", "company_policy_2.txt")
    index_path = os.path.join(PROJECT_ROOT, "faiss_preload_index")

    loader = TextLoader(data_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150

    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)

    print("Preloaded documents indexed at", index_path)


if __name__ == "__main__":
    ingest_preloaded_docs()
