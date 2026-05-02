from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np
from langchain.schema import Document

class KeywordRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = None
        self.documents = []

    def fit(self, documents: List[str]):
        """
        Fit TF-IDF on document chunks
        """
        self.documents = documents
        self.doc_vectors = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 3):
        """
        Returns top_k documents based on keyword similarity
        """
        query_vec = self.vectorizer.transform([query])

        scores = (self.doc_vectors @ query_vec.T).toarray().flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]

        #  Convert to Document objects
        results = [Document(page_content=self.documents[i]) for i in top_indices]

        return results