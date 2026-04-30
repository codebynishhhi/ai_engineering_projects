from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from langchain.schema import Document
from rag.utils.text_clean import clean_text

def build_index():
    print("Starting indexing pipeline...")

    # Initialize components
    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = EmbeddingModel()
    vector_store = VectorStore(dim=384)

    # step 1: load and clean text
    text = loader.load("data/saas_doc.pdf")
    cleaned_text = clean_text(text)

    # step 2 : create chunks 
    chunks = chunker.chunk(cleaned_text)

    # step 3 : convert chunks to Document objects
    chunk_docs = [Document(page_content=chunk) for chunk in chunks]
    print(f"Total Chunks Created:\n {len(chunk_docs)}")
    print(f"sample chunk content: \n {chunk_docs[0].page_content[:200]}")

    # step 4 : create embeddings of page content from documents of chunks created 
    embeddings = embedder.embed([doc.page_content for doc in chunk_docs])

    # step 5 : store in Faiss vector store
    vector_store.add(embeddings, chunk_docs)
    print(f"Total Documents in Vector Store:\n {len(vector_store.documents)}")

    return vector_store


if __name__ == "__main__":
    build_index()
