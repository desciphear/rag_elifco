import os
import pandas as pd
import streamlit as st

EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"

# =========================================================
# 1. Load and Cache Catalog Data
# =========================================================
@st.cache_data
def load_catalog(file_path: str) -> pd.DataFrame:
    """Loads all sheets, forward-fills merged headers, and cleans the catalog."""
    if not os.path.exists(file_path):
        st.error(f"Catalog file '{file_path}' not found.")
        st.stop()

    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    merged_columns = [
        'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
        'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'PUROLATOR'
    ]

    for sheet_name, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        available = [c for c in merged_columns if c in df.columns]
        df[available] = df[available].ffill()
        df = df.fillna("N/A")
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

df_catalog = load_catalog(EXCEL_FILE_PATH)

# =========================================================
# 2. Rule-Based Chatbot Search Engine
# =========================================================
def parse_and_search_catalog(query: str, df: pd.DataFrame):
    """
    Direct natural language search engine without LLMs.
    Matches multi-word intents (e.g., 'cabin filter for swift', 'maruti all parts', 'EK-2502').
    """
    q_clean = query.lower().strip()

    # Broad listing check
    if q_clean in ["all", "all parts", "list all", "show all", "catalog", "full catalog"]:
        return f"📋 Displaying complete catalog ({len(df)} records):", df

    # Searchable text across all relevant columns
    search_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM', 'PUROLATOR']
    combined_text = df[search_cols].astype(str).agg(' '.join, axis=1).str.lower()

    # Filter out conversational stop words
    stop_words = {
        'for', 'the', 'in', 'of', 'and', 'a', 'is', 'price', 'mrp', 'cost',
        'give', 'me', 'show', 'parts', 'part', 'filter', 'filters',
        'what', 'which', 'tell', 'all', 'any', 'every', 'list', 'please'
    }
    
    tokens = [t for t in q_clean.split() if t not in stop_words]

    # If the query only had stop words (e.g., "show all parts"), return full catalog
    if not tokens:
        return f"📋 Displaying complete catalog ({len(df)} records):", df

    # Match rows containing all non-stopword tokens
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask = mask & combined_text.str.contains(token, na=False, regex=False)

    results = df[mask]

    if results.empty:
        return (
            "❌ **No matching parts found.**\n\n"
            "Try searching by:\n"
            "* **Car Model** (e.g., `Swift`, `Alto 800`, `Ciaz`, `Baleno`)\n"
            "* **Application** (e.g., `Cabin Air`, `Oil`, `Fuel`)\n"
            "* **Part Number** (e.g., `EK-2502`, `EK-1622`)\n"
            "* **OEM Number** (e.g., `95861M74L00`)",
            None
        )

    # Distinct part number summary
    unique_parts = results['PART NO'].nunique()
    msg = f"🔍 Found **{len(results)} matching entries** across **{unique_parts} unique Part Number(s)**:"
    return msg, results

# =========================================================
# 3. Streamlit Chat UI
# =========================================================
st.set_page_config(page_title="Elofic Catalog Bot", layout="wide")
st.title("🤖 Elofic Catalog Chat Assistant")
st.caption(f"Zero-LLM Local Chat Search • {len(df_catalog)} Total Parts Loaded")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "text": "Hello! I can look up parts, prices (MRP), and OEM numbers from the Elofic catalog. What are you looking for?",
            "data": None
        }
    ]

# Display Previous Messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["data"] is not None:
            display_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'MRP', 'OEM', 'PUROLATOR']
            st.dataframe(msg["data"][[c for c in display_cols if c in msg["data"].columns]], use_container_width=True, hide_index=True)

# Chat Input Handler
if user_input := st.chat_input("Ask a question (e.g., 'All parts for Maruti', 'Swift cabin filter', 'EK-2506')..."):
    # Render user prompt
    st.session_state.chat_history.append({"role": "user", "text": user_input, "data": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Search catalog
    with st.chat_message("assistant"):
        response_text, response_df = parse_and_search_catalog(user_input, df_catalog)
        st.markdown(response_text)
        
        if response_df is not None:
            display_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'MRP', 'OEM', 'PUROLATOR']
            st.dataframe(response_df[[c for c in display_cols if c in response_df.columns]], use_container_width=True, hide_index=True)

        # Append assistant response to session state
        st.session_state.chat_history.append({
            "role": "assistant",
            "text": response_text,
            "data": response_df
        })
