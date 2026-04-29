class Retriever :

    # requires a vector store and an embedding model to be initialized
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    # You give it a question (query) and tell it how many documents you want back (top_k). It defaults to 3.
    def retrieve(self, query:str, top_k:int = 3):
        user_query_embedding = self.embedding_model.embed_query(query)
        # print(f"User query embedding: {user_query_embedding}")
        # It takes those vector numbers and asks the database, "Which 3 items in your memory are mathematically closest to these numbers?"
        return self.vector_store.search(user_query_embedding, top_k)