from rag.retriever import Retriever
from rag.llm import LLMClient


class RAGPipeline:

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = LLMClient()

    def generate_answer(self, query: str) -> str:

        # Step 1: Retrieve context
        chunks = self.retriever.retrieve(query, top_k=2)

        context = "\n\n".join(chunks)

        # Step 2: Build prompt
        prompt = f"""
        You are a helpful AI assistant and add some creativity and emoticons.

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