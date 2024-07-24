import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("koreanpopvibes.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = "Your know everything about Korean Pop! Whether they a long-time fan or new to the K-Pop world,You here to keep them updated on the latest news, discuss their favorite groups, share new music releases, and much more."
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=1000,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["How has the success of Gangnam Style influenced the global perception and reception of K-pop? "],
            ["What role does social media play in the globalization of K-pop, and how has it transformed the interaction between K-pop artists and their global fanbase?"],
            ["What factors contributed to the expansion of K-pop beyond Asia and its transformation into a global phenomenon since the mid-2000s?"],
            ["How does the rigorous training and selection process of K-pop idols contribute to the global appeal and success of K-pop artists?"],
            ["How does the intensive training and multinational composition of K-pop groups contribute to their global appeal and success?"],
            ["How do different roles within K-pop groups help maintain their popularity and stability during challenges?"],
            ["How has the use of social media, such as Twitter and YouTube, contributed to the global success of K-pop groups?"],
            ["How does social media impact the reputation management and global image of K-pop artists and entertainment companies?"]
        ],
        title='Korean-Pop Vibes🫶🫰'
    )

if __name__ == "__main__":
    demo.launch()