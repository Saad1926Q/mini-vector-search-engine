import torch
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from utils import get_embeddings
from engine import LSHIndex

EMBEDDING_PATH = Path("data/embeddings.pkl")
IMAGE_DIR = Path("data/images")
MODEL = "openai/clip-vit-base-patch32"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL)

data = get_embeddings(
    image_dir=IMAGE_DIR,
    embedding_path=EMBEDDING_PATH,
    model=model,
    processor=processor,
    device=DEVICE
)

embeddings = np.array(data["embeddings"])
image_paths = data["paths"]

NUM_HYPERPLANES = 16  
vector_dim = embeddings.shape[1]

lsh = LSHIndex(num_hyperplanes=NUM_HYPERPLANES, vector_dimension=vector_dim,num_trees=4)
lsh.build_index(dataset_vectors=embeddings,index_path="data/lsh_index.pkl")


query_vector = embeddings[0]  
top_k = 5


print(f"Searching for {image_paths[0]}")
top_k_ids = lsh.query(query_vector, k=top_k)
for rank, idx in enumerate(top_k_ids, start=1):
    print(f"{rank}: {image_paths[idx]}")
