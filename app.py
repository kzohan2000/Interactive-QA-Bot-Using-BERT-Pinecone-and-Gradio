import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
import gradio as gr
import requests

# Load model and tokenizer
model_path = "model.pth"
tokenizer_path = "tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained("bert-base-uncased")
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize Pinecone
api_key = "ef1ce0e6-971d-4119-abae-8068f14edfae"
pc = Pinecone(api_key=api_key)
index = pc.Index("genai")

class RAGModel:
    def __init__(self, index, tokenizer, model, cohere_api_key):
        self.index = index
        self.tokenizer = tokenizer
        self.model = model
        self.cohere_api_key = cohere_api_key

    def retrieve(self, question):
        query_embedding = self.encode_question(question)
        nearest_neighbors = self.index.query(vector=query_embedding.squeeze().tolist(), top_k=5)
        relevant_embeddings = [torch.tensor(res['values']) for res in nearest_neighbors['matches']]
        context_vector = self.generate_context_vector(relevant_embeddings)
        return context_vector

    def generate(self, context_vector, query):
        url = "https://api.cohere.ai/generate"
        headers = {'Authorization': f'Bearer {self.cohere_api_key}', 'Content-Type': 'application/json'}
        data = {'prompt': query, 'max_tokens': 50}
        response = requests.post(url, headers=headers, json=data)
        return response.json().get('text', "No response")

    def encode_question(self, question):
        inputs = self.tokenizer.encode_plus(question, add_special_tokens=True, max_length=512, return_attention_mask=True, return_tensors="pt", truncation=True)
        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return outputs.last_hidden_state[:, 0, :]

    def generate_context_vector(self, relevant_embeddings):
        context_vector = torch.mean(torch.stack(relevant_embeddings), dim=0)
        return context_vector

# Initialize RAG model
rag_model = RAGModel(index, tokenizer, model, "yIgpzn8UuKBEZ2qsW0Gd4OuuMvdbgMSw4432NboF")

# PDF processing
def process_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# QA System function
def qa_system(pdf, question):
    document_text = process_pdf(pdf)
    context_vector = rag_model.retrieve(question)
    response = rag_model.generate(context_vector, question)
    return response

# Gradio Interface
interface = gr.Interface(
    fn=qa_system, 
    inputs=[gr.File(type="filepath", label="Upload PDF"), gr.Textbox(label="Ask a question")], 
    outputs="text", 
    title="Interactive QA Bot",
    description="Upload a PDF document and ask questions based on its content."
)

interface.launch()
