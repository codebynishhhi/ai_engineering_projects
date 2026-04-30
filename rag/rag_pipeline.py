from rag.retriever import Retriever
from rag.llm import LLMClient
from rag.query_rewriter import QueryRewriter
from rag.multi_query_generator import MultiQueryGenerator
from rag.reranker import Reranker


class RAGPipeline:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = LLMClient()
        self.query_rewriter = QueryRewriter(self.llm)
        self.multi_query_generator = MultiQueryGenerator(self.llm)
        self.reranker = Reranker(self.llm)

    def generate_answer(self, query: str) -> str:

        # =====================================================
        # STEP 1 — Rewrite Query
        # =====================================================
        rewritten_query = self.query_rewriter.rewrite_user_query(query)
        print(f"\nRewritten Query: {rewritten_query}")

        # =====================================================
        # STEP 2 — Generate Multiple Queries
        # =====================================================
        multi_queries = self.multi_query_generator.generate_multiple_queries(rewritten_query)
        print(f"Multi Query Result: {multi_queries}")

        # =====================================================
        # STEP 3 — Retrieve Documents (IMPORTANT: returns Document objects)
        # =====================================================
        all_docs = []
        for q in multi_queries:
            docs = self.retriever.retrieve(q, top_k=3)
            all_docs.extend(docs)

        # Debug check (VERY IMPORTANT habit)
        if len(all_docs) > 0:
            print(f"Document Type Check: {type(all_docs[0])}")

        # =====================================================
        # STEP 4 — Deduplicate Documents (by content)
        # =====================================================
        seen = set()
        unique_docs = []

        for doc in all_docs:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        print(f"Unique Docs Count: {len(unique_docs)}")

        # to know which chunks were selected
        for i, doc in enumerate(unique_docs):
            print(f"===============Retrieved doc chunk {i+1}==================\n")
            print(doc.page_content[:200])

        # =====================================================
        # STEP 5 — Rerank Documents
        # =====================================================
        reranked_docs = self.reranker.rerank(rewritten_query, unique_docs)

        # =====================================================
        # STEP 6 — Build Context (convert to string ONLY here)
        # =====================================================
        context = "\n\n".join([doc.page_content for doc in reranked_docs])

        # =====================================================
        # STEP 7 — Build Prompt
        # =====================================================
        prompt = f"""
        You are a helpful AI assistant.

        You must answer ONLY using the provided context.

        Guidelines:
        - Keep answers simple, clear, and structured (bullet points)
        - Make it understandable for all age groups
        - Stay within SaaS / subscription domain
        - DO NOT introduce unrelated concepts
        - DO NOT hallucinate

        If the answer is not present, say:
        "I don't know based on the provided data."

        ---------------------
        Context:
        {context}
        ---------------------

        Question:
        {query}

        Answer:
        """

        # =====================================================
        # STEP 8 — Generate Answer
        # =====================================================
        answer = self.llm.generate_llm_response(prompt)

        return answer