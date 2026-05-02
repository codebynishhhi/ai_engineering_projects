import os
from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from langchain.schema import Document
from rag.utils.text_clean import clean_text
from rag.keyword_retriever import KeywordRetriever
from rag.utils.assign_metadata import assign_category

def build_index():
    print("Starting indexing pipeline...")

    vector_store = VectorStore(dim=384)
    keyword_retriever = KeywordRetriever()

    # =====================================================
    # 🟢 STEP 0 — LOAD EXISTING INDEX (if exists)
    # =====================================================
    if os.path.exists("artifacts/faiss.index"):
        print("🔁 Loading existing index...")

        vector_store.load("artifacts")

        # Rebuild keyword retriever (TF-IDF not persisted)
        keyword_retriever.fit([doc.page_content for doc in vector_store.documents])

        print(f"Total Documents Loaded: {len(vector_store.documents)}")

        return vector_store, keyword_retriever

    # =====================================================
    # 🟢 STEP 1 — BUILD NEW INDEX
    # =====================================================
    print("⚙️ No existing index found. Building new one...")

    loader = DocumentLoader()
    chunker = TextChunker()
    embedder = EmbeddingModel()

    # Load + clean
    text = loader.load("data/saas_doc.pdf")
    cleaned_text = clean_text(text)

    # Chunk
    chunks = chunker.chunk(cleaned_text)
        
    # Convert to Document objects
    # chunk_docs = [Document(page_content=chunk) for chunk in chunks]


    # Convert to Document objects and assign metadata category using simple keyword-based function
    chunk_docs = [
        Document(
            page_content=chunk,
            metadata = {"category": assign_category(chunk)}
        ) for chunk in chunks
    ]


    print(f"Total Chunks Created: {len(chunk_docs)}")
    print(f"Sample Chunk: {chunk_docs[0].page_content[:200]}")

    # Embeddings
    embeddings = embedder.embed([doc.page_content for doc in chunk_docs])

    # Store in FAISS
    vector_store.add(embeddings, chunk_docs)

    print(f"Total Documents in Vector Store: {len(vector_store.documents)}")

    # Fit keyword retriever
    keyword_retriever.fit([doc.page_content for doc in chunk_docs])

    # =====================================================
    # 🟢 STEP 2 — SAVE INDEX
    # =====================================================
    vector_store.save("artifacts")

    print("✅ Index built and saved successfully")

    return vector_store, keyword_retriever


if __name__ == "__main__":
    build_index()