import re

class MultiQueryGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_multiple_queries(self, user_query: str):
        prompt = f"""
        You are an expert at generating multiple search queries for a SaaS knowledge base.

        Given a user query, generate 3 variations that explore different aspects of the SAME intent.

        Rules:
        1. Do NOT change original meaning
        2. Do NOT introduce new concepts
        3. Stay within SaaS / subscription domain
        4. Keep queries concise

        User Query:
        {user_query}

        Output:
        1. ...
        2. ...
        3. ...
        """

        response = self.llm.generate_llm_response(prompt)

        queries = []
        for line in response.split("\n"):
            match = re.match(r"^\d+[\.\)]\s*(.*)", line.strip())
            if match:
                queries.append(match.group(1).strip())

        return queries