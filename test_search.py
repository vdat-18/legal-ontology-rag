import os
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181')
query = "Xe đạp không chấp hành hiệu lệnh bị phạt gì?"
embedding = model.encode([query], normalize_embeddings=True)[0].tolist()

print(f'Embedding dimension: {len(embedding)}')
print()

client = MilvusClient(uri=os.getenv('MILVUS_URI'), token=os.getenv('MILVUS_TOKEN'))
client.use_database(os.getenv('MILVUS_DATABASE'))

print('Searching...')
results = client.search(
    collection_name='naive_rag_chunks',
    data=[embedding],
    limit=5,
    output_fields=['chunk_id', 'source_name', 'text']
)

print(f'Results: {len(results[0])} hits')
if results[0]:
    for i, hit in enumerate(results[0], 1):
        dist = hit['distance']
        chunk_id = hit['entity']['chunk_id']
        source = hit['entity']['source_name']
        text = hit['entity']['text']
        print(f'{i}. distance={dist:.4f}')
        print(f'   chunk_id: {chunk_id[:32]}...')
        print(f'   source: {source}')
        print(f'   text: {text[:100]}...')
        print()

client.close()
