from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel :
    def __init__(self, model_name :str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    # creating embedding for a list of texts from chunks selected
    def embed(self, texts : List[str]) -> List[List[float]] :
        return self.model.encode(texts, show_progress_bar=True)
    
    # creating embedding for a query i.e the user input question
    # query: str: This is the specific question the user types (e.g., "What is your refund policy?").
    # [0]: Since .encode() expects a list, we wrap the query in [] and then grab the first (and only) result at index 0.
    # Why? When a user asks a question, you must convert it into the same math format as your document chunks. 
    # If the numbers for the question "match" the numbers for a document chunk, you've found the answer!
    def embed_query(self, user_query:str) -> List[float]:
        return self.model.encode([user_query])[0]