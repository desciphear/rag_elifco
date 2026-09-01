import os
import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from google import genai
from dotenv import load_dotenv
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
@st.cache_data
def load_and_clean_dataframe(file_path: str) -> pd.DataFrame:
    """Loads all sheets and applies forward fill for merged headers."""
    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    merged_columns = [
        'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
        'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'PUROLATOR'
    ]

    for sheet_name, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        available_merge_cols = [c for c in merged_columns if c in df.columns]
        df[available_merge_cols] = df[available_merge_cols].ffill()
        df = df.fillna('N/A')
        df['sheet_name'] = sheet_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_documents_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts cleaned DataFrame rows into descriptive passages."""
    documents = []
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
        sheet_name = row.get('sheet_name', 'Catalog')

        text_passage = (
            f"Elofic Part Number: {part_no} | "
            f"Vehicle Maker: {maker} | "
            f"Applicable Model: {model} | "
            f"Filter Application: {app} | "
            f"Filter Type: {part_type} | "
            f"Engine Standard: {engine_bs} | "
            f"MRP: ₹{mrp} | "
            f"Pack Size: {pack_size} | "
            f"OEM Cross-Reference: {oem} | "
            f"Purolator Cross-Reference: {purolator}"
        )

        documents.append({
            "page_content": text_passage,
            "metadata": {
                "part_no": str(part_no),
                "maker": str(maker),
                "model": str(model),
                "application": str(app),
                "mrp": str(mrp),
                "oem": str(oem),
                "sheet_name": str(sheet_name),
                "row_index": int(idx)
            }
        })
    return documents


@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initializes ChromaDB and embeds the Excel file on startup."""
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
            raise FileNotFoundError(f"Static catalog file '{EXCEL_FILE_PATH}' not found.")
        
        df_clean = load_and_clean_dataframe(EXCEL_FILE_PATH)
        docs = build_documents_from_df(df_clean)
        
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


# Load catalog dataframe & vector store
df_catalog = load_and_clean_dataframe(EXCEL_FILE_PATH)
collection = initialize_database()

# Load environment variables from a .env file if present
load_dotenv()

# Ensure GEMINI_API_KEY is provided; surface a clear error if not
_gemini_key = os.getenv("GEMINI_API_KEY")
if not _gemini_key:
    err = (
        "No GEMINI_API_KEY found. Please set the GEMINI_API_KEY environment variable. "
        "See https://ai.google.dev/gemini-api/docs/api-key for instructions."
    )
    try:
        # If running inside Streamlit, show a friendly error in the UI
        st.error(err)
    except Exception:
        pass
    raise RuntimeError(err)

client = genai.Client(api_key=_gemini_key)

# =========================================================
# 2. Hybrid Retrieval & Generation Pipeline
# =========================================================
def query_rag_pipeline(user_query: str, n_results: int = 10) -> str:
    """
    1. If user asks for all parts / catalog view, filters the full DataFrame and returns all rows.
    2. Otherwise, uses semantic vector search for specific queries.
    """
    q_lower = user_query.lower()
    is_list_all_query = any(w in q_lower for w in ["all", "every", "list", "total", "catalog", "show parts", "full"])

    # ROUTE 1: Tabular Exhaustive Query (Returns 100% of matching parts)
    if is_list_all_query:
        filtered_df = df_catalog.copy()

        # Dynamic filtering based on application
        if "cabin" in q_lower:
            filtered_df = filtered_df[filtered_df['APPLICATION'].str.upper().str.contains('CABIN', na=False)]
        elif "oil" in q_lower:
            filtered_df = filtered_df[filtered_df['APPLICATION'].str.upper().str.contains('OIL', na=False)]
        elif "fuel" in q_lower:
            filtered_df = filtered_df[filtered_df['APPLICATION'].str.upper().str.contains('FUEL', na=False)]
        elif "air" in q_lower:
            filtered_df = filtered_df[filtered_df['APPLICATION'].str.upper().str.contains('AIR', na=False)]

        # Group by Part Number so duplicates across vehicle models are aggregated cleanly
        grouped = filtered_df.groupby('PART NO').agg({
            'MAKER': 'first',
            'MODEL': lambda x: ', '.join(x.unique()),
            'APPLICATION': 'first',
            'TYPE': 'first',
            'MRP': 'first'
        }).reset_index()

        # Convert to Markdown Table directly so LLM doesn't truncate the list
        table_md = grouped[['PART NO', 'MAKER', 'APPLICATION', 'TYPE', 'MRP', 'MODEL']].to_markdown(index=False)
        return f"### 📋 Found {len(grouped)} Unique Part Numbers ({len(filtered_df)} Vehicle Applications):\n\n" + table_md

    # ROUTE 2: Semantic Vector Search for specific questions
    search_results = collection.query(
        query_texts=[user_query],
        n_results=n_results
    )
    retrieved_docs = search_results.get("documents", [[]])[0]
    context = "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No relevant parts found."

    system_instruction = (
        "You are an Elofic Auto Parts catalog assistant. "
        "Answer the question accurately using ONLY the catalog context below. "
        "Do not omit or truncate any parts present in the context. "
        "Always list Part Number, Model, Application, and MRP."
    )

    prompt_content = f"Catalog Context:\n{context}\n\nUser Question: {user_query}"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_content,
        config={
            "system_instruction": system_instruction,
            "temperature": 0.1,
        },
    )
    return response.text

# =========================================================
# 3. Streamlit UI
# =========================================================
st.set_page_config(page_title="Elofic Catalog Assistant", layout="centered")
st.title("🚗 Elofic Catalog Assistant")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_prompt := st.chat_input("Ask about Elofic parts (e.g., 'List all parts for Maruti' or 'MRP for Swift cabin filter')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalog..."):
            try:
                answer = query_rag_pipeline(user_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})