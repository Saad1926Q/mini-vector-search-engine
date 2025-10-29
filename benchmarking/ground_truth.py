"""

The purpose of this file is to compute the ground truth. 
The ground truth is the list of the actual nearest neighbors, found by a slow, brute-force search.

The reason why we are doing this is because we can't measure accuracy of our LSH implementation without knowing the ground truth values.
The good thing is that we only have to do this calculation one time.

"""

import pickle
import numpy as np
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict,Set

with open("../data/embeddings.pkl", "rb") as f:
    embedding_data = pickle.load(f)

embeddings_list = embedding_data["embeddings"]
embeddings_matrix = np.array(embeddings_list)

similarity_matrix = cosine_similarity(embeddings_matrix)

ground_truth: Dict[int, Set[int]] = {}

k=5

for i in tqdm(range(len(embeddings_list)), desc="Computing ground truth"):
    
    # We ignore the first result since it's the image itself (similarity=1.0)
    desc_indices = np.argsort(-similarity_matrix[i])
    top_k=desc_indices[1:k+1]

    ground_truth[i] = set(top_k)


with open("../data/ground_truth.pkl", "wb") as f:
    pickle.dump(ground_truth,f)