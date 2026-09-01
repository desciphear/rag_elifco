import os
import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import Any, Dict, List

# =========================================================
# Configuration
# =========================================================
EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"
COLLECTION_NAME = "elofic_catalog"
DB_PERSIST_PATH = "./elofic_vectordb"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# =========================================================
# 1. Parsing & Indexing Logic
# =========================================================
@st.cache_data
def load_and_clean_dataframe(file_path: str) -> pd.DataFrame:
    """Loads all sheets, forward-fills merged cells, and cleans the DataFrame."""
    if not os.path.exists(file_path):
        st.error(f"Catalog file '{file_path}' not found.")
        st.stop()

    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    merged_columns = [
        'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
        'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'PUROLATOR', 'Image Link'
    ]

    for sheet_name, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        available = [c for c in merged_columns if c in df.columns]
        df[available] = df[available].ffill()
        df = df.fillna("N/A")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_documents_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts rows to rich descriptive text passages for semantic search."""
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
        image_link = row.get('Image Link', 'N/A')

        passage = (
            f"Elofic Part: {part_no} | Maker: {maker} | Model: {model} | "
            f"Application: {app} | Type: {part_type} | MRP: ₹{mrp} | "
            f"OEM: {oem} | Purolator: {purolator} | Image: {image_link}"
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
                "image_link": str(image_link),
                "row_index": int(idx)
            }
        })
    return documents


@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initializes ChromaDB vector store."""
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


# Load clean catalog and vector store
df_catalog = load_and_clean_dataframe(EXCEL_FILE_PATH)
collection = initialize_database()

# OpenRouter Client
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("Please configure your `OPENROUTER_API_KEY` in Streamlit Secrets or .env file.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# =========================================================
# 2. Comprehensive Context Retriever
# =========================================================
def get_comprehensive_context(query: str) -> str:
    """
    Guarantees that ALL matching parts are retrieved:
    - If user asks broad categorical queries ('oil filter', 'cabin', 'maruti', 'all'),
      filters the entire dataset and aggregates models by Part Number.
    - If specific, uses semantic vector search.
    """
    q_lower = query.lower()

    # Detect broad categorical filters
    categories = {
        "cabin": "CABIN",
        "oil": "OIL",
        "fuel": "FUEL",
        "air": "AIR",
        "maruti": "MARUTI",
    }
    
    matched_categories = [cat_key for cat_key, col_val in categories.items() if cat_key in q_lower]
    is_broad_query = bool(matched_categories) or any(w in q_lower for w in ["all", "every", "list", "total", "catalog"])

    if is_broad_query:
        df_matched = df_catalog.copy()

        if "cabin" in q_lower:
            df_matched = df_matched[df_matched['APPLICATION'].str.upper().str.contains('CABIN', na=False)]
        elif "oil" in q_lower:
            df_matched = df_matched[df_matched['APPLICATION'].str.upper().str.contains('OIL', na=False)]
        elif "fuel" in q_lower:
            df_matched = df_matched[df_matched['APPLICATION'].str.upper().str.contains('FUEL', na=False)]
        elif "air" in q_lower:
            df_matched = df_matched[df_matched['APPLICATION'].str.upper().str.contains('AIR', na=False)]
            
        if "maruti" in q_lower:
            df_matched = df_matched[df_matched['MAKER'].str.upper().str.contains('MARUTI', na=False)]

        if not df_matched.empty:
            # Group by Part Number to consolidate models compactly for the LLM
            grouped = df_matched.groupby('PART NO').agg({
                'APPLICATION': 'first',
                'TYPE': 'first',
                'MRP': 'first',
                'MODEL': lambda x: ', '.join(x.unique()),
                'Image Link': 'first'
            }).reset_index()

            items = []
            for _, row in grouped.iterrows():
                img = f" | Image: {row['Image Link']}" if row.get('Image Link') and row.get('Image Link') != "N/A" else ""
                items.append(
                    f"- **Part No:** {row['PART NO']} | **App:** {row['APPLICATION']} | **MRP:** ₹{row['MRP']} | "
                    f"**Models:** {row['MODEL']}{img}"
                )
            return f"Found {len(grouped)} distinct Part Numbers ({len(df_matched)} vehicle applications):\n" + "\n".join(items)

    # Fallback to Semantic Vector Search for targeted queries
    search_results = collection.query(
        query_texts=[query],
        n_results=10
    )
    retrieved_docs = search_results.get("documents", [[]])[0]
    return "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No matching catalog records found."


# =========================================================
# 3. Streaming Conversational Generator
# =========================================================
def stream_conversational_rag(user_query: str):
    """Retrieves full context and streams human-like response."""
    context = get_comprehensive_context(user_query)

    system_instruction = (
        "You are an expert, helpful Elofic Auto Parts advisor. "
        "Answer naturally and conversationally using ONLY the provided catalog context.\n\n"
        "Rules:\n"
        "1. DO NOT truncate or leave out parts. If the context contains multiple parts, list all of them.\n"
        "2. DO NOT use raw Markdown tables. Use conversational paragraphs and organized bullet points with bold highlights.\n"
        "3. For each part, include Part Number, Applicable Models, Application, and MRP in ₹.\n"
        "4. If an image link exists and is not 'N/A', include it as: [View Part Image](URL).\n"
        "5. Be concise, friendly, and helpful."
    )

    prompt_content = f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"

    stream = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_content},
        ],
        temperature=0.1,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# =========================================================
# 4. Streamlit Chat Interface
# =========================================================
st.set_page_config(page_title="Elofic Parts Advisor", layout="centered")
st.title("💬 Elofic Auto Parts Advisor")
st.caption("Powered by OpenRouter • Fast, typo-tolerant conversational catalog assistant.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi there! I'm your Elofic parts advisor. Ask me anything about parts, prices, or vehicle compatibility (e.g., *'Show all oil filters'*, *'Cabin filter for Swift'*, or *'Price for Alto 800 air filter'*)."
        }
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_prompt := st.chat_input("Ask a question (e.g., 'oil filters', 'cabin filter for swfit', 'part no for ciaz')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_stream = stream_conversational_rag(user_prompt)
        full_response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
