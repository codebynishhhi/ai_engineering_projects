import re 

def clean_text(text:str) -> str:
    # Remove multiple spaces and newlines
    text = text.replace("\n", " ")
    # keep valid spaces remove extra ones 
    text = " ".join(text.split())
    text = text.strip()
    
    return text