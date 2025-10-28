import pickle
import numpy as np
from pathlib import Path
from typing import Dict,List
from numpy.typing import NDArray
from utils import hamming_distance
from sklearn.metrics.pairwise import cosine_similarity


class LSHIndex:
    def __init__(self,num_hyperplanes:int,vector_dimension:int):
        
        self.num_hyperplanes: int = num_hyperplanes
        self.vector_dimension: int = vector_dimension

        self.hyperplanes :NDArray[np.float64]=np.random.randn(num_hyperplanes,vector_dimension)

        self.hashtable :Dict[str, List[int]]={}

    def _hash(self,vector:NDArray[np.float64])->str:
        """
            Compute the binary hash string for a given vector.
            vector: shape (1, dim)

        """

        dot_pdt :NDArray[np.float64]=(self.hyperplanes@vector.T).T  # (1,num_hyperplanes)

        mask=dot_pdt>0

        hash_bits = mask.astype(int).flatten()

        hash_str = ''.join(map(str, hash_bits))

        return hash_str
    
    def build_index(self,dataset_vectors,index_path: str = None):
        """
            Build the LSH index. If index_path exists, load it instead of rebuilding.

        """
        
        if index_path and Path(index_path).exists():
            with open(index_path, 'rb') as f:
                saved_index = pickle.load(f)
                self.hyperplanes = saved_index.hyperplanes
                self.hashtable = saved_index.hashtable
                self.dataset_vectors = saved_index.dataset_vectors
            return
        
        
        self.dataset_vectors = dataset_vectors
        
        for i in range(len(dataset_vectors)):
            hash_str=self._hash(dataset_vectors[i])

            if hash_str not in self.hashtable.keys():
                self.hashtable[hash_str]=[]

            self.hashtable[hash_str].append(i)

        if index_path:
            self.save(index_path)


    def _find_closest_buckets(self,query_hash,k):
        distances=[]

        for bucket_hash in self.hashtable.keys():
            hamming_dist=hamming_distance(query_hash,bucket_hash)

            distances.append((hamming_dist,bucket_hash))

        distances.sort(key=lambda x: x[0])

        closest_buckets = [bucket_hash for _ , bucket_hash in distances[:k]]

        return closest_buckets
            

    def query(self,query_vector,k):
        """
        
        Given a query vector, return the top k results
        
        """
        
        query_hash=self._hash(query_vector)

        closest_buckets=self._find_closest_buckets(query_hash,10)

        closest_buckets=set(closest_buckets)

        cosine_similarities=[]

        for bucket in closest_buckets:
            vector_idx=self.hashtable[bucket]

            for idx in vector_idx:
                similarity=cosine_similarity(query_vector.reshape(1, -1), self.dataset_vectors[idx].reshape(1, -1))

                cosine_similarities.append((similarity,idx))

        
        cosine_similarities.sort(key=lambda x:x[0],reverse=True)

        top_k_ids = [idx for _ , idx in cosine_similarities[:k]]

        return top_k_ids

    def save(self,filepath: str):
        """
            Saves the entire LSHIndex object to a file.
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
        print("Index saved successfully to {filepath}.")
