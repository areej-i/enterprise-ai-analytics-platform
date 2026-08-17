from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_pdf(self, path):
        loader = PyPDFLoader(path)
        return loader.load()

    def chunk_documents(self, documents):
        return self.splitter.split_documents(documents)
    
    def process_pdf(self, path):
        documents = self.load_pdf(path)
        return self.chunk_documents(documents)