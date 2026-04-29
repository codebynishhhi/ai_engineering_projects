import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class LLMClient:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_llm_response(self,prompt:str) -> str:
        response = self.client.chat.completions.create(
             model="llama-3.3-70b-versatile",
             messages=[
                 {"role": "user", "content": prompt}
             ],
             temperature=0.3,
             max_tokens=500
        )
        return response.choices[0].message.content.strip()