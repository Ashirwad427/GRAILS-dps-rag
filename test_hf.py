import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    raise RuntimeError("HF_TOKEN was not found in .env")

client = InferenceClient(
    api_key=token,
    provider="auto",
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: Hugging Face connection works."
        }
    ],
    temperature=0.0,
    max_tokens=50,
)

print(response.choices[0].message.content)