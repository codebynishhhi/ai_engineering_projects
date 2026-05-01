from typing import List, Tuple
import numpy as np
from langchain.schema import Document

class Reranker:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def rerank(self, user_query : str, documents : List[Document]) -> List[Tuple[Document, float]]:
        """Reranks documents based on cosine similarity between user query and document content embeddings."""

        # get the user query embedding
        user_query_embedding = self.embedding_model.embed_query(user_query)

        scored_docs = []

        for doc in documents:
            doc_embedding = self.embedding_model.embed_query(doc.page_content)
            score = self.cosine_similarity(user_query_embedding, doc_embedding)
            scored_docs.append((doc, score))

        # sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs

