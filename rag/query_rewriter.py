class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite_user_query(self,  user_query:str) -> str:
            prompt = f""" You aare a helpful and expert assistant that rewrites user queries for better information retrieval.
            Rules - 
            - Keep original intent EXACTLY the same
            - Do NOT add new information
            - Do NOT assume anything not in the query
            - Only make it clearer and more specific
            "If the context does not explicitly mention a direct relationship,do not infer or assume connections."
            User query: {user_query}
            Rewritten query:
    
            """
            rewritten_query = self.llm.generate_llm_response(prompt)
            return rewritten_query.strip()
