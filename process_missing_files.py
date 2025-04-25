import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Define directories
SOURCE_DIR = os.path.join(os.getcwd(), "PolicyDocuments") 
OUTPUT_DIR = os.path.join(os.getcwd(), "TextFiles") 
MISSING_FILES_PATH = "missing_files.txt"  # List of missing files

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a text-based PDF using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text.strip() if text else None 

def extract_text_with_ocr(pdf_path):
    """Extract text from an image-based PDF using OCR."""
    images = convert_from_path(pdf_path)
    text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text.strip() if text else None

def process_pdf(pdf_path):
    """Process a PDF: use pdfplumber first, then OCR if needed."""
    text = extract_text_from_pdf(pdf_path)
    
    if not text:  # If no text extracted, try OCR
        print(f"Applying OCR to: {os.path.basename(pdf_path)}")
        text = extract_text_with_ocr(pdf_path)
    
    return text

def process_missing_files():
    """Process only the files listed in missing_files.txt."""
    if not os.path.exists(MISSING_FILES_PATH):
        print("Missing files list not found! Make sure missing_files.txt exists.")
        return

    # Read missing filenames from the file
    with open(MISSING_FILES_PATH, "r", encoding="utf-8") as f:
        missing_files = [line.strip() for line in f.readlines()]

    processed_count = 0

    for filename in missing_files:
        pdf_path = None
        
        # Search for the file inside PolicyDocuments
        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                if os.path.splitext(file)[0] == filename and file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    break

        if not pdf_path:
            print(f"Could not find {filename} in PolicyDocuments.")
            continue

        # Determine the relative output path
        relative_path = os.path.relpath(os.path.dirname(pdf_path), SOURCE_DIR)
        output_folder = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        output_file_path = os.path.join(output_folder, f"{filename}.txt")

        # Skip if already processed
        if os.path.exists(output_file_path):
            print(f"Skipping {filename} (already converted).")
            continue  

        # Process PDF
        extracted_text = process_pdf(pdf_path)

        # Save extracted text only if successful
        if extracted_text:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print(f"Extracted: {filename}")
            processed_count += 1
        else:
            print(f"Failed to extract: {filename}")

    print(f"\nOCR Process Complete! {processed_count} files processed.")

# Run the script
process_missing_files()
