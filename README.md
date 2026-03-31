# hybrid-search-qdrant

## Install dependencies
python3.13 -m pip install -r requirements.txt

## Run qdrant with docker
sudo docker run -d --name qdrant-test -p 6333:6333 -p 6334:6334 qdrant/qdrant

## Create points with sparse and dense vectors in qdrant
python3.13 qdrant_sparse_dense_embed.py

## Hybrid search
python3.13 hybrid_search.py