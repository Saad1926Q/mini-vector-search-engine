import pickle
import faiss
import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict,Set

with open("../data/embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

embeddings_list = embedding_data["embeddings"]
embeddings = np.array(embeddings_list)

embeddings_faiss = embeddings.astype('float32')

vector_dim=embeddings_faiss.shape[1]
NUM_HYPERPLANES = 16  

faiss_index = faiss.IndexLSH(vector_dim, NUM_HYPERPLANES)

faiss_index.add(embeddings_faiss)

faiss.write_index(faiss_index, "../data/faiss_lsh.index")