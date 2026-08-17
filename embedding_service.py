import os
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class EmbeddingService:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_vector_store(self, path="vector_db"):
        return FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def create_vector_store(self, chunks):
        return FAISS.from_documents(
            chunks,
            self.embeddings
        )
    
    def search(self, question, k=3):

        vector_store = self.load_vector_store()

        return vector_store.similarity_search(
            question,
            k=k
        )

    def save_vector_store(
        self,
        vector_store,
        path="vector_db"
    ):

        vector_store.save_local(path)