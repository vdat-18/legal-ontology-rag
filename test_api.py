import os
import requests
from dotenv import load_dotenv

load_dotenv()

base_url = os.getenv('LLM_BASE_URL')
api_key = os.getenv('LLM_API_KEY')
model = os.getenv('LLM_MODEL')

print(f'Testing API: {base_url}/chat/completions')
print(f'Model: {model}')
print(f'API Key: {api_key[:20]}...')
print()

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}

data = {
    'model': model,
    'messages': [{'role': 'user', 'content': 'Say hello in Vietnamese'}],
    'max_tokens': 50,
    'temperature': 0.0,
}

try:
    response = requests.post(f'{base_url}/chat/completions', headers=headers, json=data, timeout=10)
    print(f'Status: {response.status_code}')
    print(f'Response: {response.text[:200]}')
except Exception as e:
    print(f'Error: {e}')
