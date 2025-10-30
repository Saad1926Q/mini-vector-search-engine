import sys
import time
import faiss
import pickle
import numpy as np
from tqdm import tqdm 
from pathlib import Path
from typing import Dict,Set
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append(str(Path(__file__).parent.parent))  

from engine import LSHIndex

with open("../data/embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

with open("../data/query_embeddings.pkl", "rb") as f:
    query_embedding_data = pickle.load(f)

embeddings_list = embedding_data["embeddings"]
embeddings = np.array(embeddings_list).astype("float32")

query_embeddings_list=query_embedding_data["embeddings"]
query_embeddings=np.array(query_embeddings_list).astype("float32")

query_indices = np.arange(0,len(query_embeddings_list))

faiss_index: faiss.IndexLSH = faiss.read_index("../data/faiss_lsh.index")

with open("../data/ground_truth.pkl", "rb") as f:
    ground_truth=pickle.load(f)

NUM_HYPERPLANES = 16  
vector_dim = embeddings.shape[1]

lsh = LSHIndex.load("../data/lsh_index.pkl")


query_vector = embeddings[0]  
top_k = 5

my_latencies = []
faiss_latencies = []
bf_latencies=[]
my_recalls = []
faiss_recalls = []

for idx in tqdm(query_indices, desc="Benchmarking Queries"):

    query_vector = query_embeddings[idx]
    true_neighbors = ground_truth[idx]
    
    start_time = time.perf_counter()
    my_results = set(lsh.query(query_vector, k=top_k))
    end_time = time.perf_counter()
    my_latencies.append((end_time - start_time) * 1000) # milliseconds
    
    correct_found = len(my_results.intersection(true_neighbors))
    my_recalls.append(correct_found / top_k)

    query_vector_faiss = query_vector.reshape(1, -1)
    
    start_time = time.perf_counter()
    distances, faiss_indices = faiss_index.search(query_vector_faiss, top_k)
    end_time = time.perf_counter()
    faiss_latencies.append((end_time - start_time) * 1000)
    
    faiss_results = set(faiss_indices[0])
    correct_found = len(faiss_results.intersection(true_neighbors))
    faiss_recalls.append(correct_found / top_k)

    start_time = time.perf_counter()
    
    sims = cosine_similarity(query_vector.reshape(1, -1), embeddings)[0]

    bf_indices = np.argsort(-sims)[:top_k]
    bf_results = set(bf_indices)

    end_time = time.perf_counter()
    bf_latencies.append((end_time - start_time) * 1000)


print("\n--- Benchmark Results ---")


print("\n--- Average Latency (per query) ---")
print(f"  Brute Force Vector Search: {np.mean(bf_latencies):.4f} ms per query")
print(f"  My LSH Index: {np.mean(my_latencies):.4f} ms per query")
print(f"  FAISS LSH Index: {np.mean(faiss_latencies):.4f} ms per query")

print("\n--- Average Recall (%) ---")
print(f"  My LSH Index:  {np.mean(my_recalls) * 100:.2f}%")
print(f"  FAISS LSH Index:  {np.mean(faiss_recalls) * 100:.2f}%")