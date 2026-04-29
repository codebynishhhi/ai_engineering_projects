from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


class TextChunker:

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        chunks = self.splitter.split_text(text)
        return chunks