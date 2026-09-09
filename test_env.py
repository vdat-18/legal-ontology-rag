import os
from dotenv import load_dotenv, find_dotenv

print('Before load_dotenv:')
print(f'  LLM_API_KEY: {os.getenv("LLM_API_KEY", "NOT SET")}')

dotenv_path = find_dotenv(usecwd=True)
print(f'Dotenv path: {dotenv_path}')

if dotenv_path:
    load_dotenv(dotenv_path)
    print('After load_dotenv:')
    print(f'  LLM_API_KEY: {os.getenv("LLM_API_KEY", "NOT SET")[:20]}...')
    print(f'  LLM_BASE_URL: {os.getenv("LLM_BASE_URL", "NOT SET")}')
    print(f'  LLM_MODEL: {os.getenv("LLM_MODEL", "NOT SET")}')
else:
    print('No .env found')
