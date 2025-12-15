import json
import re
from typing import Dict, Any, List, Tuple

from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.analysis_agent import AnalysisAgent
from agents.memory_agent import MemoryAgent
from utils.llm_connector import LLMConnector
from memory.memory_models import MemoryRecord

class CoordinatorAgent(BaseAgent):
    """The Manager agent that orchestrates the workflow and makes adaptive decisions."""
    
    def __init__(self):
        super().__init__("Manager")
        
        # Initialize all worker agents
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.memory_agent = MemoryAgent()
        
        # Map agent names to instances for dynamic routing
        self.agents = {
            "ResearchAgent": self.research_agent,
            "AnalysisAgent": self.analysis_agent,
            "MemoryAgent": self.memory_agent,
        }
        
        # LLM for planning and synthesis
        self.llm_connector = LLMConnector()

    def _create_memory_record(self, topic: str, summary: str, source_refs: List[str], confidence: float) -> MemoryRecord:
        """Helper to create a MemoryRecord instance for storage."""
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        return MemoryRecord(
            memory_type="Knowledge",
            agent_source=self.name,
            topic=topic,
            summary=summary,
            source_refs=source_refs,
            confidence=confidence
        )

    def plan_workflow(self, user_query: str) -> List[Dict[str, Any]]:
        """Uses the LLM or a rule-based fallback to decompose the query into a task sequence."""
        
        system_prompt = (
            "You are an expert Task Planner. Decompose the user's complex query into a sequence of steps "
            "for specialized agents: ResearchAgent, AnalysisAgent, and MemoryAgent. "
            "Only output the JSON array, strictly following the schema. Task payload values must be strings or arrays."
            "Example Plan for 'Analyze ML techniques': [{'agent': 'ResearchAgent', 'task_payload': {'query': 'ML techniques'}}, "
            "{'agent': 'AnalysisAgent', 'task_payload': {'data': 'OUTPUT_FROM_STEP_0', 'analysis_prompt': 'Compare effectiveness'}, 'dependency': 0}]"
        )
        
        # Use LLM for planning (preferred)
        plan_json = None
        if self.llm_connector.is_functional():
            self.log.info("Generating workflow plan via LLM.")
            # First attempt: request structured JSON
            plan_json = self.llm_connector.generate_response(
                prompt=f"User Query: {user_query}", 
                system_prompt=system_prompt,
                json_mode=True
            )

            # If LLM returned non-list (malformed JSON or parsing failed), try a tolerant text parse
            if not isinstance(plan_json, list):
                self.log.warning("LLM did not return a valid JSON plan; attempting tolerant text parse.")
                text_resp = self.llm_connector.generate_response(
                    prompt=f"User Query: {user_query}",
                    system_prompt=system_prompt,
                    json_mode=False
                )

                if isinstance(text_resp, str):
                    # Try to extract a JSON array substring from the text response
                    m = re.search(r"(\[.*\])", text_resp, re.S)
                    if m:
                        try:
                            candidate = json.loads(m.group(1))
                            if isinstance(candidate, list):
                                plan_json = candidate
                                self.log.info("Parsed JSON plan from text response.")
                        except Exception:
                            self.log.debug("Failed to parse extracted JSON from LLM text response.")

        # If LLM unavailable or parsing failed, fallback to rule-based planner
        if not isinstance(plan_json, list):
            if not self.llm_connector.is_functional():
                self.log.warning("LLM unavailable. Falling back to rule-based planner.")
            else:
                self.log.warning("Falling back to rule-based planner after LLM parsing failure.")
            plan_json = self._rule_based_plan_fallback(user_query)

        return plan_json if isinstance(plan_json, list) else []

    def _rule_based_plan_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Rule-based planner for graceful degradation."""
        q_lower = query.lower()
        
        # Rule 1: Memory Test/Retrieval
        if any(w in q_lower for w in ["what did we discuss", "what did we learn", "earlier"]):
            return [{"agent": "MemoryAgent", "task_payload": {"action": "retrieve", "query": query, "search_type": "similarity"}, "dependency": None}]
        
        # Rule 2: Complex Query (Research -> Analysis)
        if any(w in q_lower for w in ["analyze", "compare", "trade-offs", "recommend"]):
            return [
                {"agent": "ResearchAgent", "task_payload": {"query": query}, "dependency": None},
                {"agent": "AnalysisAgent", "task_payload": {"data": "$OUTPUT_0", "analysis_prompt": query}, "dependency": 0}
            ]
            
        # Rule 3: Simple Query (Research Only)
        return [{"agent": "ResearchAgent", "task_payload": {"query": query}, "dependency": None}]

    def process_query(self, user_query: str) -> str:
        """The main execution loop for a user query."""
        self.log.info("--- STARTING NEW QUERY ---")
        self.log.info("User Query Received: {query}", query=user_query)
        
        # 1. Adaptive Decision: Pre-check Memory (Avoid Redundant Work)
        # Always check memory first for relevant findings
        memory_result = self.memory_agent.execute({"action": "retrieve", "query": user_query, "search_type": "similarity", "k": 1}, {})
        
        # Retrieve the MemoryRecord objects from the dictionary output
        relevant_memories = [MemoryRecord(**r) for r in memory_result.get('records', [])]
        
        # If memory is found and confidence is high, reuse it
        if relevant_memories and relevant_memories[0].confidence > 0.8: 
            best_memory = relevant_memories[0]
            self.log.warning("Memory hit! Adaptive decision: Reusing stored knowledge. Confidence: {conf}", conf=best_memory.confidence)
            
            # Update memory to track that it influenced a decision
            self.memory_agent.execute({"action": "update", "record_id": best_memory.id, "updates": {"is_used": True}}, {})
            
            self.log.info("--- QUERY COMPLETE (Memory Used) ---")
            return f"**[Answer from Memory]** Based on our previous discussion (Topic: {best_memory.topic}), we learned: {best_memory.summary}"
        
        # 2. Plan Workflow
        workflow = self.plan_workflow(user_query)
        if not workflow:
            self.log.error("Failed to generate a valid workflow plan.")
            self.log.info("--- QUERY COMPLETE (Planning Failure) ---")
            return "Error: Could not process the request (planning failed)."
        
        # 3. Execute Workflow
        step_outputs = {}
        final_result = None
        
        for i, step in enumerate(workflow):
            agent_name = step.get('agent')
            task_payload = step.get('task_payload', {})
            dependency_index = step.get('dependency')
            
            self.log.info("Step {i}: Routing to {agent}", i=i, agent=agent_name)
            
            # Resolve Dependencies ($OUTPUT_X substitution)
            if dependency_index is not None:
                # Support dependencies as single int or list of ints
                deps = []
                if isinstance(dependency_index, list):
                    # try to coerce elements to ints where possible
                    for d in dependency_index:
                        try:
                            deps.append(int(d))
                        except Exception:
                            continue
                else:
                    try:
                        deps = [int(dependency_index)]
                    except Exception:
                        deps = []

                # Filter only dependencies that refer to earlier steps
                deps = [d for d in deps if d < i]

                if deps:
                    # If AnalysisAgent, combine 'results' lists from all dependencies
                    if agent_name == "AnalysisAgent":
                        combined = []
                        for d in deps:
                            prev_output = step_outputs.get(d, {})
                            if isinstance(prev_output.get("results"), list):
                                combined.extend(prev_output.get("results"))
                        task_payload["data"] = combined
                    else:
                        # Replace any $OUTPUT_X placeholders present in the payload for each dependency
                        for d in deps:
                            placeholder = f"$OUTPUT_{d}"
                            if placeholder in str(task_payload):
                                prev_output = step_outputs.get(d, {})
                                task_payload = json.loads(json.dumps(task_payload).replace(f'"{placeholder}"', json.dumps(prev_output)))

            # Execute Agent
            agent = self.agents.get(agent_name)
            if agent:
                step_output = agent.execute(task_payload, {})
                step_outputs[i] = step_output
                final_result = step_output # The last output is the most current result
            else:
                self.log.error("Unknown agent '{agent}' in plan.", agent=agent_name)
                
        # 4. Synthesize Answer & Update Memory
        final_answer, final_summary, final_confidence, source_refs = self._synthesize_and_extract_knowledge(user_query, final_result, workflow, step_outputs)
        
        if final_summary and final_confidence > 0.5:
            # Store the final finding in the Knowledge Base
            record = self._create_memory_record(
                topic=user_query, 
                summary=final_summary, 
                source_refs=source_refs, 
                confidence=final_confidence
            )
            self.memory_agent.execute({"action": "store", "data": record.model_dump()}, {})
        
        self.log.info("--- QUERY COMPLETE ---")
        return final_answer

    def _synthesize_and_extract_knowledge(self, user_query: str, last_output: Dict[str, Any], workflow: List[Dict[str, Any]], step_outputs: Dict[int, Dict[str, Any]]) -> Tuple[str, str, float, List[str]]:
        """Synthesizes final answer using LLM and extracts a summary for storage."""
        
        analysis_result = last_output.get("analysis", last_output.get("result", "N/A"))
        final_confidence = last_output.get("confidence", 0.5)
        
        # Gather sources from the Research Agent step if it was run
        source_refs = []
        # Find the Research Agent step output by matching workflow index to step_outputs
        research_step_output = None
        for idx, step in enumerate(workflow):
            if step.get('agent') == 'ResearchAgent':
                research_step_output = step_outputs.get(idx)
                break

        if research_step_output and isinstance(research_step_output.get('results'), list):
            source_refs = [f"{r.get('source')} ({r.get('url')})" for r in research_step_output['results']]

        system_prompt = (
            "You are the final synthesis agent. Based on the user's initial query and the final analysis/research output, "
            "provide a comprehensive final answer, formatted nicely. Then, in a separate section labeled 'KNOWLEDGE_SUMMARY:', "
            "provide a concise, one-sentence summary of the key finding for long-term memory storage (vector search). "
            "Finally, state the final overall confidence score (0.0 to 1.0) on a new line labeled 'CONFIDENCE:'."
        )
        prompt = f"Original Query: {user_query}\n\nFinal Analysis/Result:\n{analysis_result}"
        
        if self.llm_connector.is_functional():
            llm_response = self.llm_connector.generate_response(prompt, system_prompt)
        else:
            # Fallback Synthesis
            llm_response = f"**[Answer Fallback]** Query: {user_query}. Result: Analysis/research complete but synthesis tool is unavailable. The core finding is contained in the output.\n\nKNOWLEDGE_SUMMARY: A simple summary based on the rule-based result.\nCONFIDENCE: 0.4"

        # Post-process LLM response
        final_answer = llm_response
        summary = ""
        confidence = final_confidence
        
        if "KNOWLEDGE_SUMMARY:" in llm_response:
            parts = llm_response.split("KNOWLEDGE_SUMMARY:")
            final_answer = parts[0].strip()
            
            summary_part = parts[1].split("CONFIDENCE:")[0].strip()
            summary = summary_part.split('\n')[0].strip() # Get the first line as the summary
        
        if "CONFIDENCE:" in llm_response:
            try:
                confidence_str = llm_response.split("CONFIDENCE:")[-1].strip()
                confidence = float(confidence_str.split('\n')[0].strip().split(' ')[0])
            except ValueError:
                self.log.warning("Could not extract confidence from synthesis.")

        return final_answer, summary, confidence, source_refs