from agents.base_agent import BaseAgent
from memory.memory_models import MemoryRecord
from memory.enhanced_memory import EnhancedMemorySystem
from typing import Dict, Any, List

class MemoryAgent(BaseAgent):
    """Manages long-term storage, retrieval, and context updates."""
    
    def __init__(self):
        super().__init__("MemoryAgent")
        # Initialize the system that manages FAISS and metadata stores
        self.memory_system = EnhancedMemorySystem()
        
    def execute(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles requests to store or retrieve knowledge.
        Input: {'action': 'store'/'retrieve', 'data': MemoryRecord / 'query': '...'}
        Output: {'records': [MemoryRecord, ...]} or {'status': 'success'}
        """
        action = task_payload.get('action')
        self.log.info("Executing memory action: {action}", action=action)
        
        if action == 'store':
            # Data must be a Pydantic object, but we accept a dict too and validate
            data = task_payload.get('data')
            if isinstance(data, dict):
                data = MemoryRecord(**data)
                
            self.memory_system.store_finding(data)
            return {"status": "success", "message": f"Stored finding for topic: {data.topic}"}
        
        elif action == 'retrieve':
            query = task_payload.get('query')
            search_type = task_payload.get('search_type', 'similarity')
            k = task_payload.get('k', 5)
            
            records: List[MemoryRecord] = self.memory_system.retrieve_knowledge(query, search_type=search_type, k=k)
            
            # The Manager needs to see the MemoryRecord data, so we convert to dicts
            return {"records": [r.model_dump(mode='json') for r in records]}
            
        elif action == 'update':
             # Used by the Manager to update flags like is_used=True
            record_id = task_payload.get('record_id')
            updates = task_payload.get('updates')
            success = self.memory_system.update_record(record_id, updates)
            return {"status": "success" if success else "error", "message": f"Update success: {success}"}

        return {"status": "error", "message": f"Unknown memory action: {action}"}