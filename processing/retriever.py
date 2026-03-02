import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SwiggyRetriever:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.index = None
        self.metadata = None

        self.load_index()
        self.load_metadata()

    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    def load_metadata(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def embed_query(self, query: str):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )
        return embedding

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.45):
        query_vector = self.embed_query(query)
        faiss.normalize_L2(query_vector)

        similarities, indices = self.index.search(query_vector, top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            similarity_score = similarities[0][i]

            if similarity_score >= similarity_threshold:
                results.append({
                    "similarity_score": float(similarity_score),
                    "chunk_text": self.metadata[idx]["text"],
                    "page_number": self.metadata[idx]["page_number"],
                    "section": self.metadata[idx]["section"]
                })

        return results


if __name__ == "__main__":
    index_path = "output/faiss_index/swiggy_faiss.index"
    metadata_path = "output/faiss_index/metadata.json"

    retriever = SwiggyRetriever(index_path, metadata_path)

    while True:
        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = retriever.search(query)

        if not results:
            print("\nNo relevant context found in the document.")
        else:
            print("\nTop Retrieved Chunks:\n")
            for i, res in enumerate(results, 1):
                print(f"Result {i}")
                print(f"Similarity Score: {res['similarity_score']:.4f}")
                print(f"Page: {res['page_number']}")
                print(f"Section: {res['section']}")
                print(f"Text Preview: {res['chunk_text'][:500]}")
                print("-" * 80)