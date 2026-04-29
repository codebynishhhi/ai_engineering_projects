from pathlib import Path
from typing import Union
from pypdf import PdfReader

class DocumentLoader:
    def load(self, file_path : Union[str, Path]) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        if file_path.suffix.lower() == '.pdf':
            return self.load_pdf(file_path)

        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self.load_text(file_path)
        
        else:
            return ValueError(f"Unsupported file type: {file_path.suffix}")

    # pdf loader
    def load_pdf(self, file_path: Path) -> str:
        reader = PdfReader(file_path)
        text = ""
        for each_page in reader.pages:
            extracted_text = each_page.extract_text()

            if extracted_text:
                text += extracted_text + "\n"
        return text
