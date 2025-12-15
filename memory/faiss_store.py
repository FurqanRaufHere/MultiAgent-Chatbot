import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Tuple
from utils.logger import log

class FAISSVectorStore:
    """
    Handles embedding generation, FAISS index management, and vector search.
    This manages the persistent knowledge store.
    """
    
    def __init__(self, index_file: str = "faiss_index.bin", mapping_file: str = "doc_map.pkl"):
        self.log = log.bind(name="FAISSVectorStore")
        
        # 1. Embedding Model Setup
        # Using a fast, standard model suitable for this project
        self.model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # 2. FAISS Index & Metadata Setup
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.doc_map = {} # Maps FAISS internal index (doc_id) to the MemoryRecord.id
        self.index = self._load_or_create_index()
        
        self.log.info("FAISS initialized with dimension: {dim}", dim=self.dimension)

    def _load_or_create_index(self):
        """Loads FAISS index and document map from disk or initializes a new one."""
        # Load map
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'rb') as f:
                self.doc_map = pickle.load(f)
            self.log.info("Loaded document map from {file}", file=self.mapping_file)

        # Load FAISS index
        if os.path.exists(self.index_file):
            self.log.info("Loading FAISS index from {file}", file=self.index_file)
            return faiss.read_index(self.index_file)
        else:
            self.log.info("Creating new FAISS IndexFlatL2 index.")
            # IndexFlatL2 is a basic index suitable for exact search and easy setup
            return faiss.IndexFlatL2(self.dimension)

    def _save_index(self):
        """Saves the FAISS index and the document map to disk."""
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'wb') as f:
            pickle.dump(self.doc_map, f)
        self.log.info("Saved FAISS index and map.")

    def get_embedding(self, text: str) -> np.ndarray:
        """Converts text into a vector embedding."""
        # The .encode method returns a numpy array
        return self.model.encode(text, convert_to_numpy=True).astype('float32').reshape(1, -1)

    def add_vector(self, record_id: str, vector: np.ndarray):
        """Adds a vector to the FAISS index and updates the document map."""
        # FAISS uses an internal incremental ID. This map links the internal ID
        # to our persistent MemoryRecord.id string.
        
        self.index.add(vector)
        internal_id = self.index.ntotal - 1
        self.doc_map[internal_id] = record_id
        self._save_index()
        self.log.debug("Added vector for record ID: {id}", id=record_id)

    def search_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches the FAISS index for the top k similar vectors.
        Returns: [(record_id, distance/score), ...]
        """
        # Ensure k is not larger than the number of vectors in the index
        k = min(self.index.ntotal, k)
        if k == 0:
            return []
            
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns -1 for empty slots, which should be ignored
            if idx != -1 and idx in self.doc_map:
                record_id = self.doc_map[idx]
                # Distances are L2, so a lower distance means higher similarity.
                # You might convert this to a score later, but we return the raw L2 distance for now.
                results.append((record_id, float(dist)))
                
        self.log.info("FAISS search returned {count} results.", count=len(results))
        return results