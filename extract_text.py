import os
import re
import pdfplumber
import docx

def extract_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    raw_text = ""
    
    # 1 & 2. Detect file type and extract text content
    if ext == '.pdf':
        raw_text = _extract_from_pdf(file_path)
    elif ext == '.docx':
        raw_text = _extract_from_docx(file_path)
    elif ext == '.txt':
        raw_text = _extract_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .pdf, .docx, or .txt.")
        
    # 3 & 4. Normalize whitespace and remove unnecessary formatting artifacts
    clean_report_text = _clean_text(raw_text)
    
    # 5. Output a single variable
    return clean_report_text

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
    """
    Normalizes whitespace and removes unwanted artifacts.
    """
    # Replace multiple spaces or tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Replace 3 or more consecutive newlines with 2 newlines (to preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove any zero-width characters or non-printable artifacts
    text = re.sub(r'[\u200b\ufeff]', '', text)
    
    # Strip leading and trailing whitespace
    return text.strip()

if __name__ == "__main__":
    clean_report_text = extract_text(r"C:\Users\Shambhavi\Desktop\forensic_dataset (1)\Addl_District_And_Sessions_Judge_vs_Jayban_Adivasi_Jay_Singh_on_6_February_2026 (1).PDF")
    print(clean_report_text)
