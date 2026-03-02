import json
import uuid
from pathlib import Path
from tqdm import tqdm
import tiktoken


class SwiggyChunker:
    def __init__(self, input_json_path: str, chunk_size: int = 800, overlap: int = 150):
        self.input_json_path = input_json_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.chunks = []

    def load_data(self):
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def count_tokens(self, text: str):
        return len(self.encoding.encode(text))

    def split_text(self, text: str):
        tokens = self.encoding.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += self.chunk_size - self.overlap

        return chunks

    def generate_chunks(self):
        data = self.load_data()

        chunk_counter = 0

        for page in tqdm(data, desc="Chunking Pages"):
            page_text = page["text"]
            page_number = page["page_number"]
            section = page["section"]

            if not page_text.strip():
                continue

            page_chunks = self.split_text(page_text)

            for chunk in page_chunks:
                chunk_data = {
                    "chunk_id": f"chunk_{chunk_counter}",
                    "text": chunk,
                    "page_number": page_number,
                    "section": section
                }

                self.chunks.append(chunk_data)
                chunk_counter += 1

    def save_chunks(self, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=4, ensure_ascii=False)

    def run(self, output_path: str):
        self.generate_chunks()
        self.save_chunks(output_path)


if __name__ == "__main__":
    input_json = "output/raw_extracted.json"
    output_json = "output/chunks.json"

    chunker = SwiggyChunker(input_json)
    chunker.run(output_json)

    print("Chunking completed successfully.")