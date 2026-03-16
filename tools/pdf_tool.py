from langchain_core.tools import tool
import os

@tool
def read_pdf(file_path: str, max_chars: int = 8000) -> str:
    """
    Extracts and returns the text content from a PDF file.
    Use this tool when a user has provided or mentioned a PDF document path.
    It extracts text page-by-page and returns a structured summary.
    
    Args:
        file_path: The absolute or relative file path to the PDF document.
        max_chars: Maximum characters to return (default 8000 to stay in context).
    
    Returns extracted text with page numbers, page count, and metadata.
    """
    try:
        import pdfplumber
    except ImportError:
        return (
            "Error: 'pdfplumber' package is not installed. "
            "Run: pip install pdfplumber"
        )
    
    if not os.path.exists(file_path):
        return (
            f"Error: File not found at '{file_path}'. "
            "Please provide the correct absolute path to the PDF file."
        )
    
    try:
        extracted_pages = []
        total_pages = 0
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    extracted_pages.append(f"--- Page {i} ---\n{text.strip()}")
        
        if not extracted_pages:
            return f"No readable text found in PDF (it may be a scanned image). Total pages: {total_pages}"
        
        full_text = "\n\n".join(extracted_pages)
        
        # Truncate if too long
        was_truncated = False
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]
            was_truncated = True
        
        header = (
            f"## PDF Content Extracted\n"
            f"**File:** {os.path.basename(file_path)}\n"
            f"**Total Pages:** {total_pages}\n"
            f"**Pages with text:** {len(extracted_pages)}\n"
            f"{'**Note:** Content truncated to first ' + str(max_chars) + ' characters.' if was_truncated else ''}\n\n"
        )
        
        return header + full_text
    
    except Exception as e:
        return f"Error reading PDF file: {str(e)}"
