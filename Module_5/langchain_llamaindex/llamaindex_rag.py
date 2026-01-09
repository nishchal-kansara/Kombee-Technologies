import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = SimpleDirectoryReader(".").load_data()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

query_engine = index.as_query_engine(
    llm=llm
)

query = "Why diwali is celebrated"
response = query_engine.query(query)

print("Question", query)
print("Answer", response)
