import requests
import json
import streamlit as st
import os

MODEL_NAME = ["nvidia/nemotron-3-super-120b-a12b:free","poolside/laguna-s-2.1:free"]

# First API call with reasoning
def get_working_model(secretKey, i):
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {secretKey}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": MODEL_NAME[i],
        "messages": [
            {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
            }
        ],
        "reasoning": {"enabled": True}
    })
    )
    if response.status_code == 200:
        return MODEL_NAME[i]
    else:
        if(i < len(MODEL_NAME) - 1):
            get_working_model(secretKey, i+1)
        else:
            return "No working model found."
