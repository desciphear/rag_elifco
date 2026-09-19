import os
import re
import time
import traceback
import requests
import pandas as pd
from fastapi import FastAPI, Request, Response
from openai import OpenAI

app = FastAPI()

# =========================================================
# Configuration
# =========================================================
EXCEL_FILE_PATH = "Data for AI Agent  19-09-2026.xls"

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
    if not os.path.exists(file_path):
        print(f"Catalog file '{file_path}' not found.")
        return pd.DataFrame()

    excel_data = pd.read_excel(file_path, sheet_name=None)
    frames = []

    for _, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(col).strip() for col in df.columns]

        # Standardize column naming variations across exports
        if 'OEM Number' in df.columns and 'OEM' not in df.columns:
            df['OEM'] = df['OEM Number']
        elif 'OEM' in df.columns and 'OEM Number' not in df.columns:
            df['OEM Number'] = df['OEM']

        if 'IMAGE LINK' in df.columns and 'Image Link' not in df.columns:
            df['Image Link'] = df['IMAGE LINK']
        elif 'Image Link' in df.columns and 'IMAGE LINK' not in df.columns:
            df['IMAGE LINK'] = df['Image Link']

        if 'Nishtha Points' not in df.columns:
            df['Nishtha Points'] = "N/A"
        if 'PACK SIZE' not in df.columns:
            df['PACK SIZE'] = "N/A"

        # Forward fill all non-empty columns to handle merged cells
        df = df.ffill().fillna("N/A")
        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

df_catalog = load_and_clean_dataframe(EXCEL_FILE_PATH)

# =========================================================
# 2. Universal Search Across All Columns
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


def get_matching_catalog_items(query: str):
    if df_catalog.empty:
        return []

    q_lower = query.lower().strip()
    (pack_op, pack_val), (pts_op, pts_val) = extract_numeric_filters(q_lower)

    df_filtered = df_catalog.copy()

    # Numerical Filter on Pack Size
    if pack_op and pack_val is not None:
        numeric_pack = pd.to_numeric(df_filtered['PACK SIZE'], errors='coerce').fillna(0)
        if pack_op == '>':
            df_filtered = df_filtered[numeric_pack > pack_val]
        elif pack_op == '<':
            df_filtered = df_filtered[numeric_pack < pack_val]
        elif pack_op == '==':
            df_filtered = df_filtered[numeric_pack == pack_val]

    # Numerical Filter on Nishtha Points
    if pts_op and pts_val is not None:
        numeric_pts = pd.to_numeric(df_filtered['Nishtha Points'], errors='coerce').fillna(0)
        if pts_op == '>':
            df_filtered = df_filtered[numeric_pts > pts_val]
        elif pts_op == '<':
            df_filtered = df_filtered[numeric_pts < pts_val]
        elif pts_op == '==':
            df_filtered = df_filtered[numeric_pts == pts_val]

    # Clean query tokens
    stop_words = {
        'get', 'all', 'where', 'for', 'the', 'in', 'of', 'and', 'filter', 'filters', 'parts', 
        'show', 'give', 'me', 'price', 'pack', 'size', 'points', 'nishtha',
        'greater', 'than', 'more', 'less', 'above', 'below', 'with', 'having',
        'is', 'are', 'what', 'which', 'can', 'you', 'find', 'item', 'items'
    }
    tokens = [t.strip() for t in q_lower.split() if t not in stop_words and not t.isdigit()]

    # Make ALL columns searchable (excluding URL links)
    excluded_cols = {'Image Link', 'IMAGE LINK'}
    search_cols = [c for c in df_filtered.columns if c not in excluded_cols]
    
    combined_series = df_filtered[search_cols].astype(str).agg(' '.join, axis=1).str.lower()
    
    mask = pd.Series(True, index=df_filtered.index)
    if tokens:
        for t in tokens:
            mask = mask & combined_series.str.contains(t, na=False, regex=False)
        df_filtered = df_filtered[mask]

    if df_filtered.empty:
        return []

    # Group by PART NO to aggregate compatibility and clean specs
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

    matching_items = []
    for _, row in grouped.iterrows():
        img_val = str(row.get('Image Link', '')).strip()
        img_url = img_val if img_val.startswith("http") else None
        
        models_display = row['MODEL'] if row['MODEL'] else 'Universal / Standard'
        pack_sz = str(row.get('PACK SIZE', 'N/A')).replace('.0', '')
        nishtha_pts = str(row.get('Nishtha Points', 'N/A')).replace('.0', '')
        oem_val = str(row.get('OEM', 'N/A')).strip()
        if oem_val in ['nan', 'None', '', 'N/A']:
            oem_val = "Not Specified"

        # Explicitly displays OEM Number on WhatsApp
        caption = (
            f"🔧 *Part No:* {row['PART NO']}\n"
            f"🏷️ *OEM:* {oem_val}\n"
            f"📦 *Pack Size:* {pack_sz} | *MRP:* ₹{row['MRP']}\n"
            f"⭐ *Nishtha Points:* {nishtha_pts}\n"
            f"⚙️ *App:* {row['APPLICATION']}\n"
            f"🚗 *Models:* {models_display}"
        )

        matching_items.append({
            "part_no": row['PART NO'],
            "caption": caption,
            "image_url": img_url
        })

    return matching_items

# =========================================================
# 3. WhatsApp Cloud API Dispatcher
# =========================================================
def send_whatsapp_text(to_number: str, text: str):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text}
    }
    return requests.post(url, headers=headers, json=payload)


def send_whatsapp_image_with_details(to_number: str, image_url: str, caption: str):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "image",
        "image": {
            "link": image_url,
            "caption": caption
        }
    }
    return requests.post(url, headers=headers, json=payload)


def dispatch_catalog_results(to_number: str, user_query: str):
    items = get_matching_catalog_items(user_query)

    if not items:
        send_whatsapp_text(to_number, "❌ No matching parts found in the Elofic catalog for your inquiry.")
        return

    total_found = len(items)
    display_limit = 5
    items_to_send = items[:display_limit]

    if total_found > display_limit:
        intro_text = (
            f"🔍 Found *{total_found}* matching parts in the catalog.\n"
            f"Showing the top *{display_limit}* results below with images:"
        )
    else:
        intro_text = f"🔍 Found *{total_found}* matching Elofic part(s):"

    send_whatsapp_text(to_number, intro_text)

    for item in items_to_send:
        time.sleep(0.3)
        if item["image_url"]:
            send_whatsapp_image_with_details(to_number, item["image_url"], item["caption"])
        else:
            send_whatsapp_text(to_number, item["caption"])

    if total_found > display_limit:
        remaining_count = total_found - display_limit
        time.sleep(0.3)
        followup_text = (
            f"📦 *+{remaining_count} more parts are available in our catalog!*\n\n"
            f"💡 To see the remaining parts or narrow down your search, please specify a vehicle model "
            f"(e.g., *'Alto K10'*, *'Swift'*), part application (e.g., *'Oil Filter'*, *'Air Filter'*), or OEM number."
        )
        send_whatsapp_text(to_number, followup_text)

# =========================================================
# 4. Webhook Endpoints
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

                dispatch_catalog_results(from_number, user_text)

    except Exception as e:
        print(f"Error processing webhook: {e}")
        traceback.print_exc()

    return Response(content="OK", status_code=200)
