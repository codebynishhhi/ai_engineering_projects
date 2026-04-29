from typing import List
import faiss
import numpy as np

# Retriever will use this vector class to find the top 3 vector chunks
class VectorStore:
    def __init__(self, dim:int):
        # the FAISS index. "FlatL2" means it stores every vector exactly as it is and uses 
        # Euclidean Distance (L2) to measure how far apart they are.
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []


    # add data chunk embeddings to the vector store database faiss
    def add(self, embeddings : List[list[float]], chunks : List[str]):
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.text_chunks.extend(chunks)

    # search for similar chunks based on the user query embedding
    def search(self, query_embedding : list[float], top_k :int= 3):
        query_vetor = np.array([query_embedding]).astype('float32')
        
        # distances, indices = self.index.search(query_vector, top_k)
        # What’s happening: This is the heavy lifting. The computer calculates the distance between your question and every single chunk you ever added to the index.
        # distances: Tells you how far away the matches are (smaller distance = more similar).
        # indices: This is the crucial part. It returns the ID numbers (the positions) of the top_k (3) closest matches. For example, it might return [5, 12, 2].

        distances, indices = self.index.search(query_vetor, top_k)
        print(f"Distances: {distances}")
        print(f"Indices: {indices}")
        results = [self.text_chunks[i] for i in indices[0] ]
        return results