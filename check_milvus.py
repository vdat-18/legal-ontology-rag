from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MilvusClient(uri=os.getenv('MILVUS_URI'), token=os.getenv('MILVUS_TOKEN'))
client.use_database(os.getenv('MILVUS_DATABASE'))

desc = client.describe_collection('naive_rag_chunks')
print('Collection: naive_rag_chunks')
print('Fields:')
for f in desc['fields']:
    print(f'  {f["name"]}: {f["type"]}')
    if f['name'] == 'vector':
        params = f.get('params', {})
        print(f'    dim: {params.get("dim", "unknown")}')
print()

print('Sample field values (first row):')
results = client.query('naive_rag_chunks', filter='', limit=1, output_fields=['embedding_profile_id', 'embedding_model_id', 'embedding_model_revision', 'chunker_config_sha256'])
if results:
    for k, v in results[0].items():
        if k != 'vector':
            print(f'  {k}: {v}')
