import pickle
import numpy as np
from pathlib import Path
from typing import Dict,List
from numpy.typing import NDArray
from utils import hamming_distance
from sklearn.metrics.pairwise import cosine_similarity


class LSHTree:
    """
    Essentially what we are doing is that our LSHIndex is going to have multiple hashtables
    LSHTree represents a single tree with random hyperplanes that hashes vectors into buckets.
    
    """
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

    
class LSHIndex:
    def __init__(self,num_hyperplanes:int,vector_dimension:int,num_trees:int):
        
        self.num_hyperplanes: int = num_hyperplanes
        self.vector_dimension: int = vector_dimension
        self.num_trees: int = num_trees

        self.trees: List[LSHTree]=[LSHTree(num_hyperplanes=self.num_hyperplanes,vector_dimension=self.vector_dimension) for _ in range(num_trees)]
    
    def build_index(self,dataset_vectors,index_path: str = None):
        """
            Build the LSH index. If index_path exists, load it instead of rebuilding.

        """        
        
        self.dataset_vectors = dataset_vectors


        for i in range(len(dataset_vectors)):
            vector=dataset_vectors[i]
            for tree in self.trees:
                hash_str=tree._hash(vector)

                if hash_str not in tree.hashtable.keys():
                    tree.hashtable[hash_str]=[]

                tree.hashtable[hash_str].append(i)

        if index_path:
            self.save(index_path)


    def query(self,query_vector,k):
        """
        
        Given a query vector, return the top k results
        
        """
        
        candidate_indices=set()

        cosine_similarities=[]

        for tree in self.trees:

            bucket=tree._hash(query_vector)

            candidate_indices.update(tree.hashtable.get(bucket, []))

        if not candidate_indices:
            # return an array of -1 of length k if no candidates found
            return np.full(k, -1, dtype=int)

        
        candidate_indices=list(candidate_indices)

        candidate_vectors=self.dataset_vectors[candidate_indices]

        query_vector=query_vector.reshape(1, -1)

        similarities = cosine_similarity(query_vector, candidate_vectors)[0]

        desc_indices = np.argsort(-similarities)

        top_k=desc_indices[:k]

        candidate_indices=np.array(candidate_indices)

        actual_top_k=candidate_indices[top_k]

        return actual_top_k

    def save(self,filepath: str):
        """
            Saves the entire LSHIndex object to a file.
        """

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
        print("Index saved successfully to {filepath}.")

    @classmethod
    def load(cls, filepath: str):
        """
            Loads a pre-built LSHIndex from a file.
        """

        with open(filepath, 'rb') as f:
            index = pickle.load(f)

        return index