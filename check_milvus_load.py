from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(uri=os.getenv('MILVUS_URI'), token=os.getenv('MILVUS_TOKEN'))
client.use_database(os.getenv('MILVUS_DATABASE'))

print('Collection load state:')
try:
    load_state = client.get_load_state('naive_rag_chunks')
    print(f'  State: {load_state}')
except Exception as e:
    print(f'  Error: {e}')

print()
print('Trying to load collection...')
try:
    client.load_collection('naive_rag_chunks')
    print('  Loaded successfully')
except Exception as e:
    print(f'  Error: {e}')

print()
print('Collection indexes:')
try:
    for idx in client.list_indexes('naive_rag_chunks'):
        print(f'  Index: {idx}')
        desc = client.describe_index('naive_rag_chunks', idx)
        print(f'    {desc}')
except Exception as e:
    print(f'  Error: {e}')
