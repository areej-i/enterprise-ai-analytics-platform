from dotenv import load_dotenv

load_dotenv()

from document_processor import DocumentProcessor
from embedding_service import EmbeddingService


processor = DocumentProcessor()

chunks = processor.process_pdf(
    "documents/employee-handbook.pdf"
)

embedding_service = EmbeddingService()
vector_store = embedding_service.create_vector_store(chunks)
embedding_service.save_vector_store(vector_store)

print("Vector database created!")