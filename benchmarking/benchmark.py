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

embeddings_list = embedding_data["embeddings"]
embeddings = np.array(embeddings_list).astype("float32")

query_indices = np.random.choice(len(embeddings), 1000, replace=False)

faiss_index: faiss.IndexLSH = faiss.read_index("../data/faiss_lsh.index")

with open("../data/ground_truth.pkl", "rb") as f:
    ground_truth=pickle.load(f)

NUM_HYPERPLANES = 16  
vector_dim = embeddings.shape[1]

lsh = LSHIndex(num_hyperplanes=NUM_HYPERPLANES, vector_dimension=vector_dim)
lsh.build_index(embeddings,index_path="../data/lsh_index.pkl")


query_vector = embeddings[0]  
top_k = 5

my_latencies = []
faiss_latencies = []
my_recalls = []
faiss_recalls = []

for idx in tqdm(query_indices, desc="Benchmarking Queries"):

    query_vector = embeddings[idx]
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


print("\n--- Benchmark Results ---")

print("\n--- My LSH Index ---")
print(f"  Average Latency: {np.mean(my_latencies):.4f} ms per query")
print(f"  Average Recall:  {np.mean(my_recalls) * 100:.2f}%")

print("\n--- FAISS LSH Index ---")
print(f"  Average Latency: {np.mean(faiss_latencies):.4f} ms per query")
print(f"  Average Recall:  {np.mean(faiss_recalls) * 100:.2f}%")