import os
import traceback
import requests
import pandas as pd
from fastapi import FastAPI, Request, Response, status
from openai import OpenAI

app = FastAPI()

# =========================================================
# Configuration
# =========================================================
EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "1298145263384348")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "elofic_secure_webhook_token_2026")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Load catalog
def load_catalog():
    if not os.path.exists(EXCEL_FILE_PATH):
        print(f"Catalog file {EXCEL_FILE_PATH} not found!")
        return pd.DataFrame()
    excel_data = pd.read_excel(EXCEL_FILE_PATH, sheet_name=None)
    frames = []
    for _, df in excel_data.items():
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        df = df.ffill().fillna("N/A")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

df_catalog = load_catalog()

# =========================================================
# Fast Catalog Retriever & LLM
# =========================================================
def search_catalog_fast(query: str):
    q = query.lower().strip()
    search_cols = [c for c in ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM'] if c in df_catalog.columns]
    combined = df_catalog[search_cols].astype(str).agg(' '.join, axis=1).str.lower()

    stop_words = {'for', 'the', 'in', 'of', 'and', 'a', 'is', 'price', 'mrp', 'cost', 'give', 'me', 'show', 'parts', 'filter', 'filters'}
    tokens = [t for t in q.split() if t not in stop_words] or q.split()

    mask = pd.Series(True, index=df_catalog.index)
    for t in tokens:
        mask = mask & combined.str.contains(t, na=False, regex=False)

    results = df_catalog[mask]
    if results.empty:
        results = df_catalog.head(5)

    image_urls, items = [], []
    for _, row in results.head(5).iterrows():
        img = str(row.get('Image Link', '')).strip()
        if img.startswith('http') and img not in image_urls:
            image_urls.append(img)
        
        items.append(
            f"- *Part No:* {row.get('PART NO', 'N/A')} | "
            f"*Model:* {row.get('MODEL', 'N/A')} | "
            f"*App:* {row.get('APPLICATION', 'N/A')} | "
            f"*MRP:* ₹{row.get('MRP', 'N/A')}"
        )

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
        max_tokens=400
    )
    return res.choices[0].message.content, image_urls

# =========================================================
# Send Message via Meta Graph API with Logging
# =========================================================
def send_meta_whatsapp_message(to_number: str, text: str, image_url: str = None):
    url = f"https://graph.facebook.com/v21.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    # Step A: Send Text Message First
    text_payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {"preview_url": False, "body": text}
    }
    r_text = requests.post(url, headers=headers, json=text_payload)
    print(f"Text Response Status: {r_text.status_code}, Body: {r_text.text}")

    # Step B: Send Image as a Separate Message if Available
    if image_url and str(image_url).startswith("http"):
        img_payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "image",
            "image": {"link": image_url.strip()}
        }
        r_img = requests.post(url, headers=headers, json=img_payload)
        print(f"Image Response Status: {r_img.status_code}, Body: {r_img.text}")

# =========================================================
# Webhook Endpoints
# =========================================================
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

        # Ignore delivery/read status updates
        if "statuses" in value and "messages" not in value:
            return Response(content="OK", status_code=200)

        messages = value.get("messages", [])
        if messages:
            msg = messages[0]
            from_number = msg.get("from")  # e.g., '919350918796'
            msg_type = msg.get("type")

            if msg_type == "text":
                user_text = msg.get("text", {}).get("body", "")
                print(f"Received query from {from_number}: {user_text}")

                bot_reply, images = get_bot_reply(user_text)
                first_img = images[0] if images else None
                send_meta_whatsapp_message(from_number, bot_reply, first_img)

    except Exception as e:
        print(f"Error processing webhook: {e}")
        traceback.print_exc()

    return Response(content="OK", status_code=200)
