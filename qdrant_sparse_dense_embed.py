from pandas import DataFrame
from fastembed import TextEmbedding, SparseTextEmbedding

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, PointStruct, SparseVector

QDRANT_IP="localhost"
QDRANT_PORT=6333
#QDRANT_API_KEY=""

# load dataset
dataset = load_dataset("ms_marco", "v1.1",split="train[:1000]")
df = DataFrame(dataset)

# extract passage texts (take the first passage text from each row)
# texts = [passages['passage_text'][0] for passages in df['passages']]
# texts = []
# for p in df['passages']:
#     selected = [t for t, s in zip(p['passage_text'], p['is_selected']) if s == 1]
#     texts.append(selected[0] if selected else p['passage_text'][0])
# passages = df['passages'].tolist()
texts = list({t for p in df['passages'] for t in p['passage_text']})

# initialize embedding models
dense_model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')
dense_embeddings = list(dense_model.embed(texts))

sparse_model = SparseTextEmbedding('Qdrant/BM25')
sparse_embeddings = list(sparse_model.embed(texts))

# create collection and insert data
try:
    # if QDRANT_API_KEY:
    #     client = QdrantClient(url=f"{QDRANT_IP}:{QDRANT_PORT}", api_key=QDRANT_API_KEY)
    # else:
    #     client = QdrantClient(host=QDRANT_IP, port=QDRANT_PORT)
    client = QdrantClient(host=QDRANT_IP, port=QDRANT_PORT)
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    exit(1)
collection_name = "hybrid_test"
dense_dim = len(dense_embeddings[0])

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

# create collection with both dense and sparse vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense_vector": VectorParams(size=dense_dim, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse_vector": SparseVectorParams()
    }
)

# insert points with both dense and sparse vectors
points = []
for i, (text, dense_vec, sparse_vec) in enumerate(zip(texts, dense_embeddings, sparse_embeddings)):
    point = PointStruct(
        id=i,
        vector={
            "dense_vector": dense_vec.tolist(),
            "sparse_vector": SparseVector(
                indices=sparse_vec.indices.tolist(),
                values=sparse_vec.values.tolist()
            )
        },
        payload={"text": text}
    )
    points.append(point)

BATCH_SIZE = 100
for batch_start in range(0, len(points), BATCH_SIZE):
    batch = points[batch_start:batch_start + BATCH_SIZE]
    client.upsert(collection_name=collection_name, points=batch)
