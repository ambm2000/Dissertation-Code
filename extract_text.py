import os
import pdfplumber
from docx import Document
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(filename="extraction_errors.log", level=logging.ERROR, format="%(asctime)s - %(message)s")

# Define correct source and output directories
SOURCE_DIR = os.path.join(os.getcwd(), "PolicyDocuments")  # Update path
OUTPUT_DIR = os.path.join(os.getcwd(), "TextFiles")  # Save text files here

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return text if text else None  # Return None for empty extractions
    except Exception as e:
        logging.error(f"Failed to extract from PDF {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extract text from a Word (.docx) file."""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text if text else None  # Return None for empty extractions
    except Exception as e:
        logging.error(f"Failed to extract from DOCX {docx_path}: {e}")
        return None

def process_files():
    """Iterate through directories and extract text from PDFs and DOCX files."""
    total_files = 0
    processed_files = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for file in tqdm(files, desc=f"Processing {root}"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, SOURCE_DIR)  # Keep folder structure
            output_folder = os.path.join(OUTPUT_DIR, relative_path)

            # Ensure output subfolder exists
            os.makedirs(output_folder, exist_ok=True)

            # Define output text file path
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.txt")

            # Skip if already processed
            if os.path.exists(output_file_path):
                continue  

            # Process PDF
            extracted_text = None
            if file.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)

            # Process DOCX
            elif file.lower().endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)

            # Skip unsupported file types
            else:
                continue

            # Save extracted text only if extraction was successful
            if extracted_text:
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                processed_files += 1

            total_files += 1

    print(f"\nExtraction complete! {processed_files}/{total_files} files successfully processed.")

# Run the script
process_files()
