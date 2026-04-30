class Reranker:
    def __init__(self, llm):
        self.llm = llm

    def rerank(self, query, documents):
        doc_texts = [doc.page_content for doc in documents]

        prompt = f"""
        You are an expert at ranking documents based on relevance.

        Query:
        {query}

        Documents:
        """

        for i, doc in enumerate(doc_texts):
            prompt += f"\nDocument {i+1}:\n{doc}\n"

        prompt += """
        Rank the documents from most relevant to least relevant.

        Return ONLY the ranked document numbers like:
        2,1,3
        """

        response = self.llm.generate_llm_response(prompt)

        try:
            order = [int(x.strip()) - 1 for x in response.split(",")]
        except:
            return documents[:3]  # fallback

        ranked_docs = [documents[i] for i in order if i < len(documents)]

        return ranked_docs[:3]