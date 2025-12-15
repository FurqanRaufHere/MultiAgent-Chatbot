import json
import os
from typing import List, Dict, Any, Optional
from memory.memory_models import MemoryRecord
from memory.faiss_store import FAISSVectorStore
from utils.logger import log

class EnhancedMemorySystem:
    """
    Manages structured persistence: FAISS for vector search and a file-based 
    Key-Value store for full MemoryRecord metadata.
    """
    
    def __init__(self, metadata_file: str = "metadata_store.json"):
        self.log = log.bind(name="EnhancedMemorySystem")
        self.vector_store = FAISSVectorStore()
        self.metadata_file = metadata_file
        self.metadata_store: Dict[str, MemoryRecord] = self._load_metadata()
        
        self.log.info("Initialized EnhancedMemorySystem with {count} records.", count=len(self.metadata_store))

    def _load_metadata(self) -> Dict[str, MemoryRecord]:
        """Loads all MemoryRecord metadata from the JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    raw_data = json.load(f)
                    # Convert raw dicts back into MemoryRecord objects
                    return {k: MemoryRecord(**v) for k, v in raw_data.items()}
            except (IOError, json.JSONDecodeError):
                self.log.error("Could not load or parse metadata file.")
                return {}
        return {}

    def _save_metadata(self):
        """Saves all MemoryRecord metadata to the JSON file."""
        try:
            # Convert MemoryRecord objects back to dictionaries for JSON serialization
            serializable_data = {k: v.model_dump(mode='json') for k, v in self.metadata_store.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.log.debug("Saved metadata store.")
        except Exception as e:
            self.log.error("Error saving metadata: {error}", error=e)


    def store_finding(self, record: MemoryRecord):
        """
        Stores a new finding in both metadata and vector stores.
        Input: a fully populated MemoryRecord object.
        """
        # 1. Generate Embedding
        vector = self.vector_store.get_embedding(record.summary)
        
        # 2. Store in FAISS
        self.vector_store.add_vector(record.id, vector)
        
        # 3. Store Metadata
        self.metadata_store[record.id] = record
        self._save_metadata()
        
        self.log.info("Successfully stored record ID: {id}", id=record.id)


    def retrieve_knowledge(self, query: str, search_type: str = 'similarity', k: int = 5) -> List[MemoryRecord]:
        """
        Retrieves records via vector similarity or keyword/topic search.
        Returns: List of relevant MemoryRecord objects.
        """
        if search_type == 'similarity':
            # 1. Get query embedding
            query_vector = self.vector_store.get_embedding(query)
            
            # 2. Search FAISS
            # Returns [(record_id, distance), ...]
            search_results = self.vector_store.search_vector(query_vector, k=k)
            
            # 3. Retrieve full records from metadata store
            records: List[MemoryRecord] = []
            for record_id, distance in search_results:
                record = self.metadata_store.get(record_id)
                if record:
                    # Optional: Augment the record with a normalized similarity score
                    # For L2 distance, we can use 1 / (1 + distance) as a simple score proxy
                    record.confidence = 1 / (1 + distance) 
                    records.append(record)
                    
            return records

        elif search_type == 'topic':
            self.log.warning("Topic search not fully implemented. Falling back to keyword match.")
            # Simple keyword search (for demonstration of topic/keyword search requirement)
            keyword_query = query.lower()
            return [
                record for record in self.metadata_store.values()
                if keyword_query in record.topic.lower() or keyword_query in record.summary.lower()
            ]

        self.log.error("Invalid search type: {type}", type=search_type)
        return []
        
    def update_record(self, record_id: str, updates: Dict[str, Any]):
        """Updates a record in the metadata store (e.g., setting is_used=True)."""
        if record_id in self.metadata_store:
            record = self.metadata_store[record_id]
            # Update the record object attributes
            for key, value in updates.items():
                if hasattr(record, key):
                    setattr(record, key, value)
                    self.log.debug("Updated record {id} field {key} to {value}", id=record_id, key=key, value=value)
            self._save_metadata()
            return True
        return False