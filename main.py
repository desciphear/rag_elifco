import os
import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from google import genai
from typing import Any, Dict, List

# =========================================================
# Configuration
# =========================================================
EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"
COLLECTION_NAME = "elofic_catalog"
DB_PERSIST_PATH = "./elofic_vectordb"

# =========================================================
# 1. Parsing & Indexing Logic
# =========================================================
def process_excel_to_documents(file_path: str) -> List[Dict[str, Any]]:
    """Loads catalog, cascades merged headers, and serializes rich context chunks."""
    excel_data = pd.read_excel(file_path, sheet_name=None)
    documents = []

    merged_columns = [
        'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
        'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'PUROLATOR', 'Image Link'
    ]

    for sheet_name, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        available = [c for c in merged_columns if c in df.columns]
        df[available] = df[available].ffill()
        df = df.fillna('N/A')

        for idx, row in df.iterrows():
            part_no = row.get('PART NO', 'N/A')
            maker = row.get('MAKER', 'N/A')
            model = row.get('MODEL', 'N/A')
            app = row.get('APPLICATION', 'N/A')
            part_type = row.get('TYPE', 'N/A')
            mrp = row.get('MRP', 'N/A')
            oem = row.get('OEM', 'N/A')
            purolator = row.get('PUROLATOR', 'N/A')
            pack_size = row.get('PACK SIZE', 'N/A')
            engine_bs = row.get('ENGINE BS', 'N/A')
            image_url = row.get('Image Link', 'N/A')

            passage = (
                f"Elofic Part Number: {part_no} | "
                f"Vehicle Maker: {maker} | "
                f"Applicable Model: {model} | "
                f"Filter Application: {app} | "
                f"Filter Type: {part_type} | "
                f"Engine Standard: {engine_bs} | "
                f"MRP: ₹{mrp} | "
                f"Pack Size: {pack_size} | "
                f"OEM Cross-Reference: {oem} | "
                f"Purolator Cross-Reference: {purolator} | "
                f"Image Link: {image_url}"
            )

            documents.append({
                "page_content": passage,
                "metadata": {
                    "part_no": str(part_no),
                    "maker": str(maker),
                    "model": str(model),
                    "application": str(app),
                    "mrp": str(mrp),
                    "oem": str(oem),
                    "image_url": str(image_url),
                    "sheet_name": sheet_name,
                    "row_index": int(idx)
                }
            })
    return documents


@st.cache_resource(show_spinner=False)
def initialize_database():
    """Builds and caches the vector database."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_client = chromadb.PersistentClient(path=DB_PERSIST_PATH)
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        if not os.path.exists(EXCEL_FILE_PATH):
            raise FileNotFoundError(f"Catalog file '{EXCEL_FILE_PATH}' not found.")
        
        docs = process_excel_to_documents(EXCEL_FILE_PATH)
        ids = [f"doc_{idx}" for idx in range(len(docs))]
        texts = [doc["page_content"] for doc in docs]
        metadatas = [doc["metadata"] for doc in docs]

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size]
            )

    return collection


collection = initialize_database()

# Load API key safely from Streamlit Secrets or Environment
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please configure your `GEMINI_API_KEY` in Streamlit Secrets or .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# =========================================================
# 2. Conversational RAG Pipeline
# =========================================================
def query_conversational_rag(user_query: str, chat_history: List[Dict[str, str]], n_results: int = 8) -> str:
    """
    1. Uses dense embeddings to retrieve records (resilient to spelling mistakes).
    2. Sends context to Gemini with conversational, human-centric formatting instructions.
    """
    # Semantic search handles typos automatically (e.g. 'swfit' -> 'Swift')
    search_results = collection.query(
        query_texts=[user_query],
        n_results=n_results
    )

    retrieved_docs = search_results.get("documents", [[]])[0]
    context = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No matching catalog entries found."

    # Conversational system prompt instructing natural text over raw tables
    system_instruction = (
        "You are an expert, helpful Elofic Auto Parts advisor. "
        "Your goal is to assist customers naturally as a knowledgeable human specialist.\n\n"
        "Guidelines:\n"
        "1. Tolerate typos, misspellings, or informal vehicle names gracefully (e.g., 'swfit' -> Swift, 'alto' -> Alto 800/K10).\n"
        "2. DO NOT output raw Markdown tables. Instead, respond in conversational prose, structured bullet points, and clear bold highlights.\n"
        "3. When presenting a part, mention the Elofic Part Number, Applicable Model, Application (Oil, Cabin Air, Fuel, etc.), and MRP in ₹.\n"
        "4. If an Image Link is available and not 'N/A', embed it as a markdown link: [View Part Image](URL).\n"
        "5. If a requested part is not present in the catalog context, politely inform the user that it is unavailable and suggest closest alternatives."
    )

    prompt_content = f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_content,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.2,
        },
    )
    return response.text

# =========================================================
# 3. Streamlit Chat Interface
# =========================================================
st.set_page_config(page_title="Elofic Parts Advisor", layout="centered")
st.title("💬 Elofic Auto Parts Advisor")
st.caption("Ask anything about parts, prices, or car compatibility in plain English.")

# Initialize message history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! I'm your Elofic parts specialist. Feel free to ask about any filter, vehicle model, or price (e.g., *'What is the price of an oil filter for Swift?'* or *'Cabin air filter for Alto 800'*)."
        }
    ]

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_prompt := st.chat_input("Ask a question (e.g., 'cabin filter for swfit', 'part no for ciaz')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up parts..."):
            try:
                answer = query_conversational_rag(user_prompt, st.session_state.messages)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Sorry, I encountered an issue retrieving that: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
