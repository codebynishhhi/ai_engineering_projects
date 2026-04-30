from rag.embedding import EmbeddingModel
from rag.retriever import Retriever
from rag.rag_pipeline import RAGPipeline
from rag.indexing import build_index

# step 1 : build index (run once)
vector_store = build_index()

# step 2 : initialize all components 
embedder = EmbeddingModel()
retriever = Retriever(vector_store, embedding_model=embedder)
rag_pipeline = RAGPipeline(retriever)

# step 3 : test with a query
query = "What about refunds?"

response = rag_pipeline.generate_answer(query)

print("\n===== FINAL ANSWER =====\n")
print(response)