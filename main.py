import os
import pandas as pd
import streamlit as st

EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"

# Columns to always display in search results
DISPLAY_COLUMNS = [
    'PART NO', 'MAKER', 'MODEL', 'APPLICATION', 
    'TYPE', 'MRP', 'OEM', 'PUROLATOR', 'Image Link'
]

# =========================================================
# 1. Load and Cache Catalog Data
# =========================================================
@st.cache_data
def load_catalog(file_path: str) -> pd.DataFrame:
    """Loads all sheets, forward-fills merged headers (including Image Link), and cleans data."""
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

df_catalog = load_catalog(EXCEL_FILE_PATH)

# =========================================================
# 2. Rule-Based Chatbot Search Engine
# =========================================================
def parse_and_search_catalog(query: str, df: pd.DataFrame):
    """Direct multi-token keyword search across all catalog columns."""
    q_clean = query.lower().strip()

    # Broad listing check
    if q_clean in ["all", "all parts", "list all", "show all", "catalog", "full catalog"]:
        return f"📋 Displaying complete catalog ({len(df)} records):", df

    search_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM', 'PUROLATOR']
    combined_text = df[search_cols].astype(str).agg(' '.join, axis=1).str.lower()

    stop_words = {
        'for', 'the', 'in', 'of', 'and', 'a', 'is', 'price', 'mrp', 'cost',
        'give', 'me', 'show', 'parts', 'part', 'filter', 'filters',
        'what', 'which', 'tell', 'all', 'any', 'every', 'list', 'please'
    }
    tokens = [t for t in q_clean.split() if t not in stop_words]

    if not tokens:
        return f"📋 Displaying complete catalog ({len(df)} records):", df

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

    unique_parts = results['PART NO'].nunique()
    msg = f"🔍 Found **{len(results)} matching entries** across **{unique_parts} unique Part Number(s)**:"
    return msg, results


def render_results_table(df_to_render: pd.DataFrame):
    """Renders the DataFrame with Image Link column as a clickable link."""
    cols_to_show = [c for c in DISPLAY_COLUMNS if c in df_to_render.columns]
    
    st.dataframe(
        df_to_render[cols_to_show],
        column_config={
            "Image Link": st.column_config.LinkColumn(
                "Image Link",
                help="Click to open part image",
                validate="^https?://",
                max_chars=40
            )
        },
        use_container_width=True,
        hide_index=True
    )

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
            "text": "Hello! I can look up parts, prices (MRP), OEM numbers, and image links from the Elofic catalog. What are you looking for?",
            "data": None
        }
    ]

# Display Previous Messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        if msg["data"] is not None:
            render_results_table(msg["data"])

# Chat Input Handler
if user_input := st.chat_input("Ask a question (e.g., 'All parts for Maruti', 'Swift cabin filter', 'EK-2506')..."):
    st.session_state.chat_history.append({"role": "user", "text": user_input, "data": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_text, response_df = parse_and_search_catalog(user_input, df_catalog)
        st.markdown(response_text)
        
        if response_df is not None:
            render_results_table(response_df)

        st.session_state.chat_history.append({
            "role": "assistant",
            "text": response_text,
            "data": response_df
        })
