import os
import re
import traceback
import requests
import pandas as pd
from fastapi import FastAPI, Request, Response
from openai import OpenAI

app = FastAPI()

# =========================================================
# Configuration
# =========================================================
DEFAULT_FILES = ["Data for AI Agent  19-09-2026.xls"]
EXCEL_FILE_PATH = next((f for f in DEFAULT_FILES if os.path.exists(f)), "Data for AI Agent  19-09-2026.xls")

META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "1298145263384348")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "elofic_secure_webhook_token_2026")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =========================================================
# 1. Parsing & Cleaning Catalog Data
# =========================================================
def load_and_clean_dataframe(file_path: str) -> pd.DataFrame:
    """Loads all sheets, normalizes column names, forward-fills merged values, and standardizes numbers."""
    if not os.path.exists(file_path):
        print(f"Catalog file '{file_path}' not found.")
        return pd.DataFrame()

    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    for _, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        # Standardize column variations across different Excel versions
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

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

df_catalog = load_and_clean_dataframe(EXCEL_FILE_PATH)

# =========================================================
# 2. Context Retrieval with Strict Numerical Logic
# =========================================================
def extract_numeric_filters(query_lower: str):
    """Detects equality and comparative filters for pack size and Nishtha points."""
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


def get_comprehensive_context(query: str):
    if df_catalog.empty:
        return "Catalog data unavailable.", []

    q_lower = query.lower()
    (pack_op, pack_val), (pts_op, pts_val) = extract_numeric_filters(q_lower)

    # Filter by Pack Size
    df_filtered = df_catalog.copy()
    if pack_op and pack_val is not None:
        numeric_pack = pd.to_numeric(df_filtered['PACK SIZE'], errors='coerce').fillna(0)
        if pack_op == '>':
            df_filtered = df_filtered[numeric_pack > pack_val]
        elif pack_op == '<':
            df_filtered = df_filtered[numeric_pack < pack_val]
        elif pack_op == '==':
            df_filtered = df_filtered[numeric_pack == pack_val]

    # Filter by Nishtha Points
    if pts_op and pts_val is not None:
        numeric_pts = pd.to_numeric(df_filtered['Nishtha Points'], errors='coerce').fillna(0)
        if pts_op == '>':
            df_filtered = df_filtered[numeric_pts > pts_val]
        elif pts_op == '<':
            df_filtered = df_filtered[numeric_pts < pts_val]
        elif pts_op == '==':
            df_filtered = df_filtered[numeric_pts == pts_val]

    # Keyword Search (for vehicle models, makers, parts, etc.)
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

    if df_filtered.empty:
        return "No matching parts found matching the specified criteria in the catalog.", []

    # Group by PART NO to prevent duplicates
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
    unique_images = []
    # Include up to 25 items so large lists aren't truncated
    for _, row in grouped.head(25).iterrows():
        img_val = str(row.get('Image Link', '')).strip()
        has_image = img_val.startswith("http")
        if has_image and img_val not in unique_images:
            unique_images.append(img_val)

        img_str = f" | Image: {img_val}" if has_image else " | Image: N/A"
        models_display = row['MODEL'] if row['MODEL'] else 'Universal / Standard'
        pack_sz = str(row.get('PACK SIZE', 'N/A')).replace('.0', '')
        nishtha_pts = str(row.get('Nishtha Points', 'N/A')).replace('.0', '')

        items.append(
            f"- *Part No:* {row['PART NO']} | *Pack Size:* {pack_sz} | *MRP:* ₹{row['MRP']} | "
            f"*Nishtha Points:* {nishtha_pts} | *App:* {row['APPLICATION']} | *Models:* {models_display}{img_str}"
        )

    context_str = f"Found {len(grouped)} matching Part Numbers:\n" + "\n".join(items)
    return context_str, unique_images

# =========================================================
# 3. Conversational Generator (WhatsApp-tailored)
# =========================================================
def get_bot_reply(user_query: str):
    context, image_urls = get_comprehensive_context(user_query)

    if context.startswith("No matching parts found"):
        return context, []

    system_instruction = (
        "You are an expert Elofic Auto Parts advisor on WhatsApp.\n\n"
        "FORMATTING RULES:\n"
        "1. Answer concisely using WhatsApp Markdown (*bold* with single asterisks, NEVER double asterisks **).\n"
        "2. List ALL parts present in the Catalog Context that match the customer's request. Do not arbitrarily skip parts.\n"
        "3. For each part, include: Part Number, Pack Size, MRP in ₹, Loyalty/Nishtha Points, Application, and Compatible Models.\n"
        "4. DO NOT output markdown image tags like ![img](url).\n"
        "5. Be direct, accurate, and professional."
    )

    res = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"}
        ],
        temperature=0.1,
        max_tokens=1500
    )
    
    reply_text = res.choices[0].message.content
    reply_text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', reply_text)
    return reply_text, image_urls

# =========================================================
# 4. WhatsApp Cloud API Dispatcher
# =========================================================
def send_meta_whatsapp_message(to_number: str, text: str, image_urls: list):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    # Step A: Send Text Response
    text_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": True, "body": text}
    }
    r_text = requests.post(url, headers=headers, json=text_payload)
    print(f"Text Response Status: {r_text.status_code}")

    # Step B: Send at most one distinct preview image
    if image_urls:
        first_img = image_urls[0]
        if str(first_img).startswith("http"):
            img_payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "image",
                "image": {"link": first_img}
            }
            r_img = requests.post(url, headers=headers, json=img_payload)
            print(f"Image Response Status: {r_img.status_code}")

# =========================================================
# 5. Webhook Endpoints
# =========================================================
@app.get("/")
@app.get("/health")
async def health_check():
    return {"status": "alive", "service": "elofic-whatsapp-bot"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=str(params.get("hub.challenge")), media_type="text/plain", status_code=200)
    return Response(content="Verification failed", media_type="text/plain", status_code=403)

@app.post("/webhook")
async def handle_meta_message(request: Request):
    data = await request.json()

    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "statuses" in value and "messages" not in value:
            return Response(content="OK", status_code=200)

        messages = value.get("messages", [])
        if messages:
            msg = messages[0]
            from_number = msg.get("from")
            msg_type = msg.get("type")

            if msg_type == "text":
                user_text = msg.get("text", {}).get("body", "")
                print(f"Received query from {from_number}: {user_text}")

                bot_reply, images = get_bot_reply(user_text)
                send_meta_whatsapp_message(from_number, bot_reply, images)

    except Exception as e:
        print(f"Error processing webhook: {e}")
        traceback.print_exc()

    return Response(content="OK", status_code=200)
