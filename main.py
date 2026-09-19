import os
import re
import chromadb
import pandas as pd
import streamlit as st
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import Any, Dict, List

# =========================================================
# Configuration
# =========================================================
DEFAULT_FILES = ["Data for AI Agent  19-09-2026.xls"]
EXCEL_FILE_PATH = next((f for f in DEFAULT_FILES if os.path.exists(f)), "Data for AI Agent  19-09-2026.xls")

COLLECTION_NAME = "elofic_catalog_v2"
DB_PERSIST_PATH = "./elofic_vectordb"
OPENROUTER_MODEL = "google/gemini-2.5-flash"

# =========================================================
# 1. Parsing & Indexing Logic
# =========================================================
@st.cache_data
def load_and_clean_dataframe(file_path: str) -> pd.DataFrame:
    """Loads all sheets, normalizes columns, forward-fills merged cells, and cleans DataFrame."""
    if not os.path.exists(file_path):
        st.error(f"Catalog file '{file_path}' not found.")
        st.stop()

    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    for _, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        if 'OEM Number' in df.columns and 'OEM' not in df.columns:
            df['OEM'] = df['OEM Number']
        if 'IMAGE LINK' in df.columns and 'Image Link' not in df.columns:
            df['Image Link'] = df['IMAGE LINK']
        if 'Nishtha Points' not in df.columns:
            df['Nishtha Points'] = "N/A"
        if 'PACK SIZE' not in df.columns:
            df['PACK SIZE'] = "N/A"

        merged_columns = [
            'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
            'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'Nishtha Points',
            'OEM', 'PUROLATOR', 'MAHLE', 'SOFIMA', 'BOSCH', 'Image Link'
        ]

        available = [c for c in merged_columns if c in df.columns]
        df[available] = df[available].ffill()
        df = df.fillna("N/A")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def build_documents_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Converts rows to descriptive text passages for semantic search."""
    documents = []
    for idx, row in df.iterrows():
        part_no = row.get('PART NO', 'N/A')
        maker = row.get('MAKER', 'N/A')
        model = row.get('MODEL', 'N/A')
        app = row.get('APPLICATION', 'N/A')
        part_type = row.get('TYPE', 'N/A')
        mrp = row.get('MRP', 'N/A')
        pack_size = str(row.get('PACK SIZE', 'N/A')).replace('.0', '')
        nishtha_pts = str(row.get('Nishtha Points', 'N/A')).replace('.0', '')
        oem = row.get('OEM', 'N/A')
        purolator = row.get('PUROLATOR', 'N/A')
        image_link = row.get('Image Link', 'N/A')

        passage = (
            f"Elofic Part: {part_no} | Maker: {maker} | Model: {model} | "
            f"Application: {app} | Type: {part_type} | MRP: ₹{mrp} | "
            f"Pack Size: {pack_size} | Nishtha Loyalty Points: {nishtha_pts} | "
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
                "pack_size": str(pack_size),
                "nishtha_points": str(nishtha_pts),
                "oem": str(oem),
                "image_link": str(image_link),
                "row_index": int(idx)
            }
        })
    return documents

@st.cache_resource(show_spinner=False)
def initialize_database():
    """Initializes ChromaDB vector store with the versioned collection."""
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

df_catalog = load_and_clean_dataframe(EXCEL_FILE_PATH)
collection = initialize_database()

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
def extract_numeric_filters(query_lower: str):
    pack_op, pack_val = None, None
    m_gt = re.search(r'(?:pack\s*size|pack)\s*(?:is\s*)?(?:>|>=|greater than|more than|above|over)\s*(\d+)', query_lower)
    m_lt = re.search(r'(?:pack\s*size|pack)\s*(?:is\s*)?(?:<|<=|less than|under|below)\s*(\d+)', query_lower)
    m_eq = re.search(r'(?:pack\s*size|pack)\s*(?:is\s*|equals?\s*|=|:\s*)?(\d+)', query_lower)

    if m_gt:
        pack_op, pack_val = '>', int(m_gt.group(1))
    elif m_lt:
        pack_op, pack_val = '<', int(m_lt.group(1))
    elif m_eq:
        pack_op, pack_val = '==', int(m_eq.group(1))

    pts_op, pts_val = None, None
    p_gt = re.search(r'(?:nishtha\s*points?|points?)\s*(?:is\s*)?(?:>|>=|greater than|more than|above|over)\s*(\d+)', query_lower)
    p_lt = re.search(r'(?:nishtha\s*points?|points?)\s*(?:is\s*)?(?:<|<=|less than|under|below)\s*(\d+)', query_lower)
    p_eq = re.search(r'(?:nishtha\s*points?|points?)\s*(?:is\s*|equals?\s*|=|:\s*)?(\d+)', query_lower)

    if p_gt:
        pts_op, pts_val = '>', int(p_gt.group(1))
    elif p_lt:
        pts_op, pts_val = '<', int(p_lt.group(1))
    elif p_eq:
        pts_op, pts_val = '==', int(p_eq.group(1))

    return (pack_op, pack_val), (pts_op, pts_val)


def get_comprehensive_context(query: str) -> str:
    q_lower = query.lower()
    (pack_op, pack_val), (pts_op, pts_val) = extract_numeric_filters(q_lower)

    df_filtered = df_catalog.copy()
    if pack_op and pack_val is not None:
        numeric_pack = pd.to_numeric(df_filtered['PACK SIZE'], errors='coerce').fillna(0)
        if pack_op == '>':
            df_filtered = df_filtered[numeric_pack > pack_val]
        elif pack_op == '<':
            df_filtered = df_filtered[numeric_pack < pack_val]
        elif pack_op == '==':
            df_filtered = df_filtered[numeric_pack == pack_val]

    if pts_op and pts_val is not None:
        numeric_pts = pd.to_numeric(df_filtered['Nishtha Points'], errors='coerce').fillna(0)
        if pts_op == '>':
            df_filtered = df_filtered[numeric_pts > pts_val]
        elif pts_op == '<':
            df_filtered = df_filtered[numeric_pts < pts_val]
        elif pts_op == '==':
            df_filtered = df_filtered[numeric_pts == pts_val]

    stop_words = {
        'get', 'all', 'where', 'for', 'the', 'in', 'of', 'and', 'filter', 'filters', 'parts', 
        'show', 'give', 'me', 'price', 'pack', 'size', 'points', 'nishtha',
        'greater', 'than', 'more', 'less', 'above', 'below', 'with', 'having',
        'is', 'are', 'what', 'which', 'can', 'you', 'find', 'item', 'items'
    }
    tokens = [t.strip() for t in q_lower.split() if t not in stop_words and not t.isdigit()]

    if tokens:
        search_cols = [c for c in ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM', 'PUROLATOR', 'MAHLE', 'BOSCH'] if c in df_filtered.columns]
        combined_series = df_filtered[search_cols].astype(str).agg(' '.join, axis=1).str.lower()
        mask = pd.Series(True, index=df_filtered.index)
        for t in tokens:
            mask = mask & combined_series.str.contains(t, na=False, regex=False)
        df_filtered = df_filtered[mask]

    if not df_filtered.empty:
        grouped = df_filtered.groupby('PART NO').agg({
            'APPLICATION': 'first',
            'TYPE': 'first',
            'MRP': 'first',
            'PACK SIZE': 'first',
            'Nishtha Points': 'first',
            'MODEL': lambda x: ', '.join(dict.fromkeys(str(v) for v in x if str(v) != 'N/A')),
            'OEM': 'first',
            'Image Link': 'first'
        }).reset_index()

        items = []
        for _, row in grouped.head(25).iterrows():
            img_val = str(row.get('Image Link', '')).strip()
            img_str = f" | Image: {img_val}" if img_val.startswith("http") else " | Image: N/A"
            models_display = row['MODEL'] if row['MODEL'] else 'Universal / Standard'
            pack_sz = str(row.get('PACK SIZE', 'N/A')).replace('.0', '')
            pts = str(row.get('Nishtha Points', 'N/A')).replace('.0', '')

            items.append(
                f"- **Part No:** **{row['PART NO']}** | **Pack Size:** {pack_sz} | **MRP:** ₹{row['MRP']} | "
                f"**Nishtha Points:** {pts} | **App:** {row['APPLICATION']} | "
                f"**Models:** {models_display}{img_str}"
            )
        return f"Found {len(grouped)} matching Part Numbers:\n" + "\n".join(items)

    search_results = collection.query(query_texts=[query], n_results=10)
    retrieved_docs = search_results.get("documents", [[]])[0]
    return "\n".join(f"- {doc}" for doc in retrieved_docs) if retrieved_docs else "No matching catalog records found."

# =========================================================
# 3. Streaming Conversational Generator
# =========================================================
def stream_conversational_rag(user_query: str):
    context = get_comprehensive_context(user_query)

    system_instruction = (
        "You are an expert, helpful Elofic Auto Parts advisor.\n\n"
        "CRITICAL RULES:\n"
        "1. DO NOT truncate or omit matching parts from the context. Present all parts returned in the Catalog Context.\n"
        "2. MANDATORY IMAGE RENDERING: For EVERY part that has an Image URL (starting with http), you MUST render it inline immediately below the part details using Markdown format: ![Part Preview](URL).\n"
        "3. DO NOT use Markdown tables. Use clean bullet points with bold highlights.\n"
        "4. For each part, include: Part Number, Pack Size, MRP in ₹, Nishtha Points, Application, OEM, Compatible Models, and the rendered image.\n"
        "5. If a part has no valid image link (or is 'N/A'), omit the image markdown for that part.\n"
        "6. Answer accurately based on the catalog context."
    )

    prompt_content = f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"

    stream = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_content},
        ],
        temperature=0.1,
        max_tokens=2500,
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
            "content": "Hi there! I'm your Elofic Parts Advisor. We have filters for 2W, 3W, Cars, LCV-HCV, Tractors & Earthmovers. Ask me anything about our filters, prices, pack sizes, loyalty points, or compatibility."
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question (e.g., 'parts where pack size is 100', 'parts where pack size is greater than 100')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_stream = stream_conversational_rag(user_prompt)
        full_response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
