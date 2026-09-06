"""Drop incompatible Milvus collection."""
from pymilvus import connections, utility

connections.connect(host="localhost", port="19530")

if utility.has_collection("naive_rag_chunks"):
    utility.drop_collection("naive_rag_chunks")
    print("✅ Collection 'naive_rag_chunks' dropped successfully")
else:
    print("ℹ️ Collection 'naive_rag_chunks' does not exist")

connections.disconnect("default")
