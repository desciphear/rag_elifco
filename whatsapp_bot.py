import os
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Form
from fastapi.responses import Response
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

# 1. Load Catalog & Persistent DB
EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"
DB_PERSIST_PATH = "./elofic_vectordb"
COLLECTION_NAME = "elofic_catalog"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
chroma_client = chromadb.PersistentClient(path=DB_PERSIST_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# 2. OpenRouter Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def query_catalog(query: str) -> str:
    search_results = collection.query(query_texts=[query], n_results=6)
    retrieved_docs = search_results.get("documents", [[]])[0]
    context = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No matching parts found."

    system_instruction = (
        "You are an expert Elofic Auto Parts advisor assisting over WhatsApp. "
        "Answer concisely in friendly plain text. Use WhatsApp formatting (*bold* with single asterisks). "
        "List Part Number, Models, Application, and MRP in ₹. "
        "Render image links as plain clickable URLs if available."
    )

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Catalog Context:\n{context}\n\nCustomer Inquiry: {query}"},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    return response.choices[0].message.content

@app.get("/")
def health_check():
    return {"status": "active", "service": "Elofic WhatsApp Bot"}

@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...)):
    bot_reply = query_catalog(Body.strip())
    twiml = MessagingResponse()
    twiml.message(bot_reply)
    return Response(content=str(twiml), media_type="application/xml")
