import pickle
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm 
import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Any



def generate_embeddings(image_paths: List[Path], model: CLIPModel, processor: CLIPProcessor,device:torch.device) -> Dict[str, Any]:
    all_embeddings = []
    all_paths_str = []

    batch_size = 32

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(path) for path in batch_paths]

        try:
            inputs = processor(
                text=None, 
                images=images, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            with torch.no_grad():
                image_features = model.get_image_features(**inputs)

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            all_embeddings.extend(image_features.cpu().numpy())
            all_paths_str.extend([str(path) for path in batch_paths])



        except Exception as e:
            print(f"Error processing batch {i // batch_size}: {e}")
            continue

    
    embedding_data = {
        "paths": all_paths_str,
        "embeddings": [emb.astype(np.float64) for emb in all_embeddings] 
    }

    return embedding_data



def save_embeddings(data: Dict[str, Any],embedding_path:Path):
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(embedding_path, "wb") as f:
        pickle.dump(data, f)

    print("Embeddings saved.")



def get_embeddings(image_dir: Path,embedding_path: Path,model: CLIPModel,processor: CLIPProcessor,device:torch.device) -> Dict[str, Any]:
     """
        Tries to load embeddings from the pickle file.
        If not found, generates them from the image directory and saves them.

    """
     
     try:
        with open(embedding_path, "rb") as f:
            embedding_data = pickle.load(f)

        return embedding_data
     
     except:
         
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        image_paths = [
            p for p in image_dir.rglob("*") 
            if p.suffix.lower() in image_extensions
        ]

        if not image_paths:
            print(f"No images found in {image_dir}. Please add images to this directory.")

        embedding_data = generate_embeddings(image_paths, model, processor,device)

        save_embeddings(embedding_data,embedding_path)
        
        return embedding_data
     

     
