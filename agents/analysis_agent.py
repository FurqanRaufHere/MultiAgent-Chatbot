from agents.base_agent import BaseAgent
from utils.llm_connector import LLMConnector
from typing import Dict, Any

class AnalysisAgent(BaseAgent):
    """Performs comparisons, reasoning, and simple calculations on retrieved data."""
    
    def __init__(self):
        super().__init__("AnalysisAgent")
        # Initialize LLM connector here, anticipating its full implementation in Phase 4
        self.llm_connector = LLMConnector() 
        
    def execute(self, task_payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input: {'data': [research_output], 'analysis_prompt': 'Compare their efficiency and summarize trade-offs.'}
        Output: {'analysis': 'Transformer architecture is more efficient...', 'confidence': 0.9}
        """
        data = task_payload.get('data', [])
        analysis_prompt = task_payload.get('analysis_prompt', 'Analyze the provided data.')
        
        self.log.info("Preparing analysis task: Analyzing {data_len} data sources.", data_len=len(data))
        
        if not data:
            self.log.warning("Analysis failed: No data provided to analyze.")
            return {"analysis": "N/A", "confidence": 0.1, "error": "No data to analyze."}

        # Format the data for the LLM
        formatted_data = "\n--- Snippet ---\n".join([item.get('snippet', '') for item in data])
        
        system_prompt = (
            "You are a specialized AI analysis agent. Your task is to perform reasoning, comparison, "
            "and summarization based *only* on the provided data. Do not use external knowledge. "
            "Conclude your analysis with a final confidence score (0.0 to 1.0) on a new line labeled 'CONFIDENCE:'."
        )
        user_prompt = f"DATA TO ANALYZE:\n{formatted_data}\n\nANALYSIS INSTRUCTION: {analysis_prompt}"
        
        llm_response = self.llm_connector.generate_response(
            prompt=user_prompt, 
            system_prompt=system_prompt,
            json_mode=False
        )

        if llm_response:
            # Attempt to extract confidence score
            confidence = 0.8
            if 'CONFIDENCE:' in llm_response:
                try:
                    # Extract the numerical value after the CONFIDENCE: tag
                    confidence = float(llm_response.split('CONFIDENCE:')[-1].strip().split('\n')[0].split(' ')[0])
                except ValueError:
                    self.log.warning("Could not extract confidence from LLM analysis. Defaulting to 0.8.")
                
            self.log.info("Analysis complete with confidence: {conf}", conf=confidence)
            return {"analysis": llm_response, "confidence": confidence}
        else:
            # Rule-based fallback if LLM is unavailable
            fallback_analysis = f"Analysis failed (LLM unavailable). Rule-based summary: Data analysis could not be performed on {len(data)} snippets."
            self.log.warning("Analysis fallback used.")
            return {"analysis": fallback_analysis, "confidence": 0.4, "error": "LLM failed/unavailable."}