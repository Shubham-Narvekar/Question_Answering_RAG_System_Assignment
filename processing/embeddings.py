import json
import numpy as np
import faiss
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SwiggyVectorStore:
    def __init__(self, chunks_path: str, output_dir: str):
        self.chunks_path = chunks_path
        self.output_dir = output_dir
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.index = None
        self.metadata = []

    def load_chunks(self):
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_embeddings(self, chunks):
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def build_faiss_index(self, embeddings):
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        self.index = index

    def save_index(self):
        os.makedirs(self.output_dir, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(self.output_dir, "swiggy_faiss.index")
        )

        with open(os.path.join(self.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, ensure_ascii=False)

    def run(self):
        chunks = self.load_chunks()

        embeddings = self.create_embeddings(chunks)

        self.build_faiss_index(embeddings)

        self.metadata = chunks

        self.save_index()


if __name__ == "__main__":
    chunks_path = "output/chunks.json"
    output_dir = "output/faiss_index"

    vector_store = SwiggyVectorStore(chunks_path, output_dir)
    vector_store.run()

    print("FAISS index built and saved successfully.")