from rag.embedding import EmbeddingModel
from rag.retriever import Retriever
from rag.rag_pipeline import RAGPipeline
from rag.indexing import build_index
from rag.eval import evaluate


# step 1 : build index (run once)
vector_store, keyword_retriever = build_index()
# step 2 : initialize all components 
embedder = EmbeddingModel()
retriever = Retriever(vector_store, embedding_model=embedder)
rag_pipeline = RAGPipeline(retriever, keyword_retriever)

# step 3 : test with a query

query = "refund for failed payment"

response = rag_pipeline.generate_answer(query)
evaluate(rag_pipeline)

print("\n===== FINAL ANSWER =====\n")
print(response)