import fitz  # PyMuPDF
import json
import re
from tqdm import tqdm
from pathlib import Path


class SwiggyPDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.document = None
        self.extracted_data = []

    def load_pdf(self):
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")
        self.document = fitz.open(self.pdf_path)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def detect_section_heading(self, text: str):
        patterns = [
            r'BOARD.?S REPORT',
            r'FINANCIAL SUMMARY',
            r'Corporate information',
            r'Organisation structure',
            r'CHANGE IN NAME',
            r'AUDITORS.? REPORT',
            r'CONSOLIDATED FINANCIAL STATEMENT',
            r'STANDALONE FINANCIAL STATEMENT'
        ]

        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return pattern
        return None

    def extract(self):
        self.load_pdf()

        current_section = "Unknown"

        for page_number in tqdm(range(len(self.document)), desc="Extracting Pages"):
            page = self.document[page_number]
            text = page.get_text("text")

            if not text.strip():
                continue

            cleaned_text = self.clean_text(text)

            detected_section = self.detect_section_heading(cleaned_text)
            if detected_section:
                current_section = detected_section

            page_data = {
                "page_number": page_number + 1,
                "section": current_section,
                "text": cleaned_text
            }

            self.extracted_data.append(page_data)

    def save_to_json(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.extracted_data, f, indent=4, ensure_ascii=False)

    def run(self, output_path: str):
        self.extract()
        self.save_to_json(output_path)


if __name__ == "__main__":
    pdf_path = "data/Annual-Report-FY-2023-24 (1) (1).pdf"
    output_path = "output/raw_extracted.json"

    extractor = SwiggyPDFExtractor(pdf_path)
    extractor.run(output_path)

    print("PDF extraction completed successfully.")