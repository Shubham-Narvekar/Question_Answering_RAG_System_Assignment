import os
from dotenv import load_dotenv
from google import genai
from retriever import SwiggyRetriever

# Load environment variables from .env
load_dotenv()


class SwiggyRAG:
    def __init__(self):
        # Load Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")

        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)

        # Use stable public Gemini model
        self.model_name = "gemini-3-flash-preview"

        # Initialize FAISS retriever
        self.retriever = SwiggyRetriever(
            "output/faiss_index/swiggy_faiss.index",
            "output/faiss_index/metadata.json"
        )

    def build_prompt(self, query, retrieved_chunks):
        """
        Build strict grounded prompt for Gemini.
        """

        context_text = ""

        for i, chunk in enumerate(retrieved_chunks):
            context_text += (
                f"[Source {i+1} | Page {chunk['page_number']}]\n"
                f"{chunk['chunk_text']}\n\n"
            )

        prompt = f"""
You are a strict financial document question-answering assistant.

IMPORTANT RULES:
- Use ONLY the provided context.
- Extract the exact number or fact from the context.
- Do NOT summarize unnecessarily.
- Do NOT truncate the answer.
- If a number exists, state it clearly.
- Always mention the page number.
- If the answer is not explicitly present, respond exactly with:
  "The answer is not available in the provided document."

Context:
{context_text}

Question:
{query}

Provide a precise and complete answer:
"""
        return prompt

    def answer(self, query):
        """
        Execute full RAG pipeline:
        Retrieval → Prompt → Gemini → Answer
        """

        # Retrieve top 3 relevant chunks
        retrieved_chunks = self.retriever.search(
            query,
            top_k=3,
            similarity_threshold=0.40
        )

        if not retrieved_chunks:
            return "The answer is not available in the provided document."

        # Build prompt
        prompt = self.build_prompt(query, retrieved_chunks)

        # Call Gemini
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": 0.0,
                "top_p": 1.0,
                "max_output_tokens": 800,
            }
        )

        return response.text


if __name__ == "__main__":
    rag = SwiggyRAG()

    
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        answer = rag.answer(question)

        print("\nAnswer:\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")