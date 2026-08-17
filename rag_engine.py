from embedding_service import EmbeddingService


class RAGEngine:

    def __init__(self):
        self.embedding_service = EmbeddingService()

    def retrieve_context(self, question, k=5):

        documents = self.embedding_service.search(question, k=k)

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        return context