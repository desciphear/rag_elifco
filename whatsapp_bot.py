import os
import requests
import pandas as pd
from fastapi import FastAPI, Request, Response, Query
from openai import OpenAI

app = FastAPI()

# =========================================================
# Configuration & Environment Variables
# =========================================================
EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "elofic_secure_webhook_token_2026")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =========================================================
# 1. Load Catalog
# =========================================================
def load_catalog():
    if not os.path.exists(EXCEL_FILE_PATH):
        return pd.DataFrame()
    excel_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
    frames = []
    merged_cols = [
        'PART NO', 'MAKER', 'SEGMENT', 'APPLICATION',
        'TYPE', 'ENGINE BS', 'PACK SIZE', 'MRP', 'PUROLATOR', 'Image Link'
    ]
    for _, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        avail = [c for c in merged_cols if c in df.columns]
        df[avail] = df[avail].ffill()
        df = df.fillna("N/A")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

df_catalog = load_catalog()

# =========================================================
# 2. Fast Catalog Retriever & LLM Call
# =========================================================
def search_catalog_fast(query: str):
    q = query.lower().strip()
    search_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM', 'PUROLATOR']
    combined = df_catalog[search_cols].astype(str).agg(' '.join, axis=1).str.lower()

    stop_words = {'for', 'the', 'in', 'of', 'and', 'a', 'is', 'price', 'mrp', 'cost', 'give', 'me', 'show', 'parts', 'filter', 'filters'}
    tokens = [t for t in q.split() if t not in stop_words] or q.split()

    mask = pd.Series(True, index=df_catalog.index)
    for t in tokens:
        mask = mask & combined.str.contains(t, na=False, regex=False)

    results = df_catalog[mask]
    if results.empty:
        results = df_catalog.head(6)

    image_urls, items = [], []
    for _, row in results.head(6).iterrows():
        img = row.get('Image Link', '')
        if img and str(img).startswith('http') and img not in image_urls:
            image_urls.append(img)
        items.append(f"- *Part No:* {row['PART NO']} | *Model:* {row['MODEL']} | *App:* {row['APPLICATION']} | *MRP:* ₹{row['MRP']}")

    return "\n".join(items), image_urls

def get_bot_reply(user_query: str):
    context, image_urls = search_catalog_fast(user_query)
    system_instruction = (
        "You are an expert Elofic Auto Parts advisor on WhatsApp. "
        "Answer concisely with WhatsApp styling (*bold* with single asterisks). "
        "List Part Number, Compatible Models, Application, and MRP in ₹."
    )
    res = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"}
        ],
        temperature=0.1,
        max_tokens=500
    )
    return res.choices[0].message.content, image_urls

# =========================================================
# 3. Send Message via Meta Graph API
# =========================================================
def send_meta_whatsapp_message(to_number: str, text: str, image_url: str = None):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    if image_url:
        # Send image with text caption
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "image",
            "image": {
                "link": image_url,
                "caption": text[:1024]  # Meta WhatsApp caption limit
            }
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {"preview_url": False, "body": text}
        }

    requests.post(url, headers=headers, json=payload)

# =========================================================
# 4. Meta Webhook Endpoints
# =========================================================
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token")
):
    """Meta webhook verification challenge"""
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain")
    return Response(content="Verification failed", status_code=403)

@app.post("/webhook")
async def handle_meta_message(request: Request):
    """Receives WhatsApp JSON payloads directly from Meta"""
    data = await request.json()
    
    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        messages = value.get("messages", [])

        if messages:
            msg = messages[0]
            from_number = msg.get("from")  # e.g., '918178580614'
            user_text = msg.get("text", {}).get("body", "")

            if user_text and from_number:
                bot_text, images = get_bot_reply(user_text)
                img_url = images[0] if images else None
                send_meta_whatsapp_message(from_number, bot_text, img_url)

    except Exception as e:
        print(f"Error handling message: {e}")

    return Response(content="OK", status_code=200)
