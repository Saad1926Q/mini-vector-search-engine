"""

The purpose of this file is to compute the ground truth. 
The ground truth is the list of the actual nearest neighbors, found by a slow, brute-force search.

The reason why we are doing this is because we can't measure accuracy of our LSH implementation without knowing the ground truth values.
The good thing is that we only have to do this calculation one time.

"""
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from typing import Dict,Set

sys.path.append(str(Path(__file__).parent.parent)) 

from utils import get_embeddings



QUERY_DIR =Path("../data/images/benchmark_queries")
QUERY_EMBEDDING_PATH = Path("../data/query_embeddings.pkl")
MODEL = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL)


with open("../data/embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

embeddings_list = embedding_data["embeddings"]
embeddings_matrix = np.array(embeddings_list)


query_data=get_embeddings(
    image_dir=QUERY_DIR,
    embedding_path=QUERY_EMBEDDING_PATH,
    model=model,
    processor=processor,
    device=DEVICE
)

query_embeddings_list=query_data["embeddings"]
query_embeddings_matrix=np.array(query_embeddings_list).astype("float32")



similarity_matrix = cosine_similarity(query_embeddings_matrix,embeddings_matrix)

ground_truth: Dict[int, Set[int]] = {}

k=5

for i in tqdm(range(len(query_embeddings_list)), desc="Computing ground truth"):
    
    desc_indices = np.argsort(-similarity_matrix[i])
    top_k=desc_indices[:k]

    ground_truth[i] = set(top_k)


with open("../data/ground_truth.pkl", "wb") as f:
    pickle.dump(ground_truth,f)