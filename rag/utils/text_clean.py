import re 

def clean_text(text:str) -> str:
    # Remove multiple spaces and newlines
    text = re.sub(r'\n+', '\n', text)

    text = re.sub(r"\s+", '', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text