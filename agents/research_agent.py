from agents.base_agent import BaseAgent
import json
import os
from typing import Dict, Any, List, Optional

class ResearchAgent(BaseAgent):
    """Simulates information retrieval using a pre-loaded knowledge base (mock search)."""
    
    def __init__(self):
        super().__init__("ResearchAgent")
        self.knowledge_base = self._load_knowledge_base()
        self.log.info("Loaded mock knowledge base with {count} topics.", count=len(self.knowledge_base))

    def _load_knowledge_base(self) -> Dict[str, List[Dict[str, str]]]:
        """Loads mock search data from the JSON file."""
        # Find the path dynamically relative to the script location
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'knowledge_base.json')
        try:
            with open(data_path, 'r') as f:
                return {k.lower(): v for k, v in json.load(f).items()} # Store keys in lower case for easier matching
        except FileNotFoundError:
            self.log.error("Knowledge base file not found at: {path}", path=data_path)
            return {}
        except json.JSONDecodeError:
            self.log.error("Knowledge base file is corrupted (invalid JSON).")
            return {}

    def _find_best_match(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Simple keyword matching against the knowledge base keys."""
        query_lower = query.lower()
        
        # Priority 1: Exact Key Match
        if query_lower in self.knowledge_base:
            return self.knowledge_base[query_lower]

        # Priority 2: Partial/Fuzzy Match (Simple containment)
        best_match_key = None
        for key in self.knowledge_base.keys():
            if query_lower in key or key in query_lower:
                best_match_key = key
                break
        
        if best_match_key:
            self.log.debug("Found partial match for query: '{query}' in key: '{key}'", query=query, key=best_match_key)
            return self.knowledge_base[best_match_key]
        
        return None

    def execute(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: {'query': 'Find recent papers on reinforcement learning'}
        Output: {'results': [{'source': 'Mock KB', 'snippet': '...'}, ...], 'confidence': 0.8}
        """
        query = task_payload.get('query', '')
        self.log.info("Simulating research for query: {query}", query=query)
        
        results = self._find_best_match(query)
        
        if results:
            self.log.info("Research successful. Retrieved {count} snippets.", count=len(results))
            # Assign a high confidence for successful retrieval from a known mock source
            return {"results": results, "confidence": 0.9} 
        else:
            self.log.warning("Research failed. No relevant knowledge found in mock base.")
            return {"results": [], "confidence": 0.0, "error": "No matching data found."}