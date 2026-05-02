from typing import List
import faiss
import numpy as np
from langchain.schema import Document
import os
import pickle

class VectorStore:
    def __init__(self, dim: int):
        """
        dim: embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
        """
        # faiss index for efficient similarity search
        self.index = faiss.IndexFlatL2(dim)
        self.documents: List[Document] = []  # store full documents

    # =====================================================
    # ADD DOCUMENTS + EMBEDDINGS
    # =====================================================
    def add(self, embeddings: List[List[float]], documents: List[Document]):
        """
        embeddings: List of embedding vectors
        documents: List of Document objects (same order as embeddings)
        """
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")

        vectors = np.array(embeddings).astype("float32")

        self.index.add(vectors)
        self.documents.extend(documents)

    # =====================================================
    # SEARCH SIMILAR DOCUMENTS
    # =====================================================
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Document]:
        """
        Returns top_k most similar Document objects
        """
        if len(self.documents) == 0:
            raise ValueError("Vector store is empty. Add documents before searching.")

        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        print(f"\nDistances: {distances}")
        print(f"Indices: {indices}")

        # indices shape: [[i1, i2, i3]]
        results = []

        for i in indices[0]:
            if i < len(self.documents):  # safety check
                results.append(self.documents[i])

        return results
    
    # =====================================================
    # SAVE INDEX + DOCUMENTS
    # =====================================================

    def save(self, path :str = "artifacts"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Vector store saved successfully!")

    # ===================================================== 
    # LOAD INDEX + DOCUMENTS
    # =====================================================

    def load(self, path : str = "artifacts"):
        if not os.path.exists(f"{path}/faiss.index") or not os.path.exists(f"{path}/documents.pkl"):
            raise FileNotFoundError("Index or documents not found. Please check the path.")

        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        print(f"Vector store loaded successfully!")