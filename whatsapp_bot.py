import os
import re
import pandas as pd
from fastapi import FastAPI, Form
from fastapi.responses import Response
from openai import OpenAI
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()

EXCEL_FILE_PATH = "Elofic AI Agent Data.xlsx"

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
# 2. OpenRouter Client
# =========================================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def search_catalog_and_retrieve(query: str):
    """Searches catalog and extracts matched rows + valid image URLs."""
    q = query.lower().strip()
    search_cols = ['PART NO', 'MAKER', 'MODEL', 'APPLICATION', 'TYPE', 'OEM', 'PUROLATOR']
    combined = df_catalog[search_cols].astype(str).agg(' '.join, axis=1).str.lower()

    stop_words = {'for', 'the', 'in', 'of', 'and', 'a', 'is', 'price', 'mrp', 'cost', 'give', 'me', 'show', 'parts', 'filter', 'filters'}
    tokens = [t for t in q.split() if t not in stop_words]
    
    if not tokens:
        tokens = q.split()

    mask = pd.Series(True, index=df_catalog.index)
    for t in tokens:
        mask = mask & combined.str.contains(t, na=False, regex=False)

    results = df_catalog[mask]
    if results.empty:
        results = df_catalog.head(8)

    # Extract valid image URLs to attach as media
    image_urls = []
    items = []
    for _, row in results.head(6).iterrows():
        img_url = row.get('Image Link', '')
        if img_url and str(img_url).startswith('http') and img_url not in image_urls:
            image_urls.append(img_url)
            
        items.append(
            f"- *Part No:* {row['PART NO']} | *Model:* {row['MODEL']} | "
            f"*App:* {row['APPLICATION']} | *MRP:* ₹{row['MRP']}"
        )
    
    return "\n".join(items), image_urls


def query_llm(user_query: str):
    context, image_urls = search_catalog_and_retrieve(user_query)
    
    system_instruction = (
        "You are an expert Elofic Auto Parts advisor on WhatsApp. "
        "Answer concisely using WhatsApp formatting (*bold* with single asterisks). "
        "List Part Number, Applicable Models, Application, and MRP in ₹. "
        "Keep it conversational and friendly. Do not write markdown image syntax."
    )

    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Catalog Context:\n{context}\n\nCustomer Inquiry: {user_query}"},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    
    text_reply = response.choices[0].message.content
    return text_reply, image_urls


# =========================================================
# 3. Webhook with Media Attachment
# =========================================================
@app.get("/")
def health_check():
    return {"status": "ok", "loaded_rows": len(df_catalog)}

@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(default="")):
    bot_reply, image_urls = query_llm(Body.strip())
    
    twiml = MessagingResponse()
    msg = twiml.message()
    msg.body(bot_reply)  # Explicitly set text body

    # Only attach media if a valid public image URL exists
    if image_urls and str(image_urls[0]).startswith("http"):
        msg.media(str(image_urls[0]).strip())

    return Response(content=str(twiml), media_type="application/xml")
