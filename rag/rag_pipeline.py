from rag.retriever import Retriever
from rag.llm import LLMClient
from rag.query_rewriter import QueryRewriter


class RAGPipeline:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = LLMClient()
        self.query_rewriter = QueryRewriter(self.llm)

    def generate_answer(self, query: str) -> str:

        # Step 0: Rewrite the user query
        rewritten_query = self.query_rewriter.rewrite_user_query(query)
        print(f"Rewritten Query: {rewritten_query}")
        # Step 1: Retrieve context
        chunks = self.retriever.retrieve(rewritten_query, top_k=2)

        context = "\n\n".join(chunks)

        # Step 2: Build prompt
        prompt = f"""
        You are a helpful AI assistant and add some creativity to help all ages of users undrstand the answer.
        You have access to the following information from a knowledge base. 
        Use this information only from the knowledge base to answer the question at the end.
        Keep it point wise and concise and improve the answer so that its easier to understand for users of all ages.
        Answer the question ONLY using the context below.
        If the answer is not present, say "I don't know based on the provided data."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # Step 3: Call LLM
        answer = self.llm.generate_llm_response(prompt)

        return answer