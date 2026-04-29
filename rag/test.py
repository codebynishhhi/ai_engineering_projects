from rag.loader import DocumentLoader
from rag.chunker import TextChunker
from rag.utils.text_clean import clean_text
from rag.embedding import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.rag_pipeline import RAGPipeline

loader = DocumentLoader()
chunker = TextChunker()
embedder = EmbeddingModel()
vector_store = VectorStore(dim=384)
retriever = Retriever(vector_store, embedding_model=embedder)
rag_pipeline = RAGPipeline(retriever)

# load and clean text
text = loader.load("data/saas_doc.pdf")
cleaned_text = clean_text(text)

# create chunks
chunks = chunker.chunk(text)

# create embeddings of created chunks and add to vector store
embeddings = embedder.embed(chunks)
vector_store.add(embeddings, chunks)


# Query
query = "what about refunds?"

response = rag_pipeline.generate_answer(query)

print("\n===== FINAL ANSWER =====\n")
print(response)

# print(f"Total Chunks: {len(chunks)}")   

# print("\n--- SECOND CHUNK ---")
# print(chunks[1])

# print(f"Embedding dimension: {len(embeddings[0])}")

# # Query
# query = "Explain refund rules and conditions"
# query_embedding = embedder.embed_query(query)
# results = vector_store.search(query_embedding, top_k=2)

# print("\n-------User query-------\n")
# print(f"User Query: {query}")

# print("\n-------Top 3 relevant chunks-------")
# for i, res in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(res)
