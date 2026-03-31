from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FusionQuery, Fusion, Prefetch, SparseVector

# QDRANT_IP="http://localhost"
QDRANT_IP="localhost"
QDRANT_PORT=6333
QDRANT_API_KEY=""

query_text = input("Enter your query: ")

# embed query locally
dense_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
query_dense = list(dense_model.embed([query_text]))[0].tolist()

sparse_model = SparseTextEmbedding('Qdrant/BM25')
query_sparse_raw = list(sparse_model.embed([query_text]))[0]
query_sparse = SparseVector(
    indices=query_sparse_raw.indices.tolist(),
    values=query_sparse_raw.values.tolist()
)

# get qdrant client
try:
    # client = QdrantClient(url=f"{QDRANT_IP}:{QDRANT_PORT}", api_key=QDRANT_API_KEY)
    client = QdrantClient(host=QDRANT_IP, port=QDRANT_PORT)
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    exit(1)
collection_name = "hybrid_test"

results = client.query_points(
    collection_name=collection_name,
    query=FusionQuery(fusion=Fusion.RRF),
    prefetch=[
        Prefetch(
            query=query_dense,
            using="dense_vector",
        ),
        Prefetch(
            query=query_sparse,
            using="sparse_vector",
        ),
    ],
    limit=10,
)

for point in results.points:
    print(f"Score: {point.score:.4f} | {point.payload['text']}")