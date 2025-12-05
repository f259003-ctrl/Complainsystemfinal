from pypdf import PdfReader

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        text += content + "\n" if content else ""
    return text

def chunk_text(text, chunk_size=800):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
