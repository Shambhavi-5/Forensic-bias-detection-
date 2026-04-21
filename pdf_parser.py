import os
import re
import pdfplumber
import docx
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def extract_sentences(file_path: str) -> list:
    """Extracts text from PDF/DOCX and splits into sentences."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        raw_text = _extract_from_pdf(file_path)
    elif ext == '.docx':
        raw_text = _extract_from_docx(file_path)
    elif ext == '.txt':
        raw_text = _extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")
        
    clean_text = _clean_text(raw_text)
    
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(clean_text)
    
    # Filter very short or invalid sentences
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]
    return valid_sentences

def _extract_from_pdf(file_path: str) -> str:
    extracted_pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
    return "\n".join(extracted_pages)

def _extract_from_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def _extract_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()
    
def _clean_text(text: str) -> str:
    # Normalize whitespaces to maintain neat sentences
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove weird hyphens that split words across lines
    text = re.sub(r'-\n+', '', text)
    text = re.sub(r'\n{2,}', ' . ', text) # Add dot to force sentence break between large paragraphs
    text = text.replace('\n', ' ')
    text = re.sub(r'[\u200b\ufeff]', '', text)
    return text.strip()

if __name__ == "__main__":
    pass
