# KRR Multi-Agent Chatbot

A research-focused multi-agent chatbot that demonstrates a Knowledge Representation & Retrieval (KRR) workflow: decomposing user queries into specialized agent tasks (research, analysis, memory), synthesizing answers, and storing/retrieving long-term knowledge using a vector store (FAISS).

This README explains what the project does, how it works, how to run it, common troubleshooting steps, and how to extend it. It focuses on practical usage and architecture rather than a plain file listing.

---
## Key Features

- Agent orchestration via a `CoordinatorAgent` that plans workflows and routes tasks to specialized agents.
- Specialized worker agents:
  - `ResearchAgent` — fetches or simulates information sources.
  - `AnalysisAgent` — analyzes research outputs and produces structured analysis.
  - `MemoryAgent` — handles persistent storage and retrieval (FAISS-backed vector store).
- LLM-based planning and synthesis through a pluggable connector (`utils/llm_connector.py`) with graceful fallback to rule-based planning.
- FAISS vector store integration for semantic memory (embedding generation via SentenceTransformers).
- CLI scripts to run automated scenario tests and an interactive runtime REPL chat.

## Architecture Overview (high-level)

- Coordinator (`agents/coordinator.py`): central manager. Receives the user query, checks memory first, plans a workflow (LLM or rule-based fallback), routes steps to agents, aggregates outputs, synthesizes a final answer, and decides whether to store a knowledge summary in memory.

- Agents:
  - ResearchAgent: simulates or performs data retrieval (mock knowledge base is used by default for offline testing).
  - AnalysisAgent: applies reasoning/analysis to Research outputs.
  - MemoryAgent: wraps vector store (FAISS) + persistent metadata and provides `store`, `retrieve`, `update` actions.

- LLM Connector (`utils/llm_connector.py`): thin wrapper around Groq client that provides `generate_response(prompt, system_prompt, json_mode=False)` and a `is_functional()` check so the system can gracefully fall back.

- FAISS Vector Store (`memory/faiss_store.py`): embeddings via SentenceTransformers, FAISS for nearest-neighbor search, and persistent mapping between internal FAISS ids and memory records.

## Behavior Notes

- Memory-first: the coordinator always queries memory (similarity search) before launching expensive workflows. If a memory entry has high confidence (>0.8), it's reused.
- Planning: the coordinator prefers the LLM for decomposing tasks. If the model returns malformed or non-JSON output, the planner attempts to extract JSON from text, otherwise falls back to a rule-based planner.
- Execution: workflow steps include optional `dependency` fields. Dependency values may be an int or list of step indices; the coordinator resolves `$OUTPUT_X` placeholders and combines outputs when needed (e.g., for the `AnalysisAgent`).
- Synthesis: final answer synthesis is performed by the LLM (if available), which should include a `KNOWLEDGE_SUMMARY:` and `CONFIDENCE:` markers to enable automatic memory storage.

## Where to Look When Something Fails (Troubleshooting)

- Groq API key missing / LLM not working
  - Symptom: Logs show `GROQ_API_KEY not found. LLM functionality is DISABLED.` or `Failed to initialize Groq client`.
  - Fix: Ensure `.env` contains `GROQ_API_KEY` or export the env var. Validate with `python scripts/check_groq_api_key.py`.

- `ModuleNotFoundError: No module named 'agents'` when running scripts
  - Fix: Run scripts from project root or use the provided `scripts/*` which prepend project root to `sys.path`. Ensure your current working directory is the repository root when invoking scripts.

- FAISS / embedding issues
  - Symptom: errors during import or when creating index.
  - Fix: Verify `sentence-transformers` and `faiss` packages are installed in the same Python environment. If FAISS wheel doesn't match your Python version/architecture, install a compatible wheel or use CPU-only packages like `faiss-cpu`.

- Planning failures (empty workflow)
  - Symptom: Output shows `Failed to generate a valid workflow plan.` followed by planning failure.
  - Fix: If LLM responses are malformed, the coordinator will try a tolerant parse. If that still fails, the rule-based planner runs. Review logs in the `outputs/` files to see what the connector returned and adjust `system_prompt` or `utils/llm_connector.py` error handling.

## Testing & Debugging

- Automated scenarios: `python main.py` writes human-readable logs to `outputs/` for each scenario. Inspect those files to see step-by-step logs.
- LLM health test: `python scripts/check_groq_api_key.py --question "What is 2+2?"`.
- Interactive: `python scripts/chat.py` for free-form testing.

## Extending the System

- Add new agents: Implement a new agent class in `agents/` following the `BaseAgent` API, then register it in the `CoordinatorAgent`'s `self.agents` map and update planning/system prompts if needed.
- Replace or mock the LLM: `utils/llm_connector.py` is intentionally small. Swap in another provider or add a local mocking/testing mode that returns deterministic plans/responses.
- Improve dependency resolution: the planner currently supports `$OUTPUT_{i}` placeholder replacement and list-of-dependencies. For more complex graphs, add an explicit DAG resolver.
- Improve memory management: add TTLs, namespaces, or stronger source validation before storing entries.

## Safety & Costs

- LLM calls may incur usage costs. The project includes rule-based fallbacks to allow offline testing and reduce calls.
- Avoid committing your `.env` or API keys to source control. Add `.env` to `.gitignore`.

## Scripts & What They Do

- **`python main.py`**: Run the predefined automated test scenarios.
  - What it does: executes a suite of scenarios (Simple Query, Complex Query, Memory Test step 1 & 2, Multi-step, Collaborative), logs progress, and writes each scenario's detailed run output to `outputs/<scenario_filename>.txt`.
  - Use when you want batch-run examples and to produce artifact files for inspection or regression testing.

- **`python scripts/chat.py`**: Start an interactive console-based chat REPL.
  - What it does: boots the `CoordinatorAgent` and accepts runtime user questions. For each input it plans/executions a workflow and prints the system's response immediately.
  - Use when you want live conversation with the multi-agent system and immediate answers.

Tips:
- Inspect the `outputs/` directory after running `main.py` to view scenario logs and diagnose planning/execution failures.
- Use `scripts/chat.py` for experimentation and quick iteration; use `main.py` to produce reproducible outputs for comparison or unit tests.
