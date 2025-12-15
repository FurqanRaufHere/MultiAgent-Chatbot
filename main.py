import sys
import os
import contextlib
import time

# Ensure all necessary classes are imported
from agents.coordinator import CoordinatorAgent
from utils.logger import setup_logger, log 

# Define the five required scenarios and their corresponding output files
SCENARIOS = {
    "Simple Query": {
        "query": "What are the main types of neural networks?",
        "file": "simple_query.txt"
    },
    "Complex Query": {
        "query": "Research transformer architectures, analyze their computational efficiency, and summarize key trade-offs.",
        "file": "complex_query.txt"
    },
    # Note: Memory Test is split into two steps for demonstration
    "Memory Test (Step 1: Store)": {
        "query": "What are the main types of neural networks?",
        "file": "memory_test_step1_store.txt"
    },
    "Memory Test (Step 2: Reuse)": {
        "query": "What did we discuss about neural networks earlier?",
        "file": "memory_test_step2_reuse.txt"
    },
    "Multi-step": {
        "query": "Find recent papers on reinforcement learning, analyze their methodologies, and identify common challenges.",
        "file": "multi_step.txt"
    },
    "Collaborative": {
        "query": "Compare two machine-learning approaches (CNN vs RNN) and recommend which is better for image classification.",
        "file": "collaborative.txt"
    }
}

OUTPUTS_DIR = "outputs"

# Context manager to redirect sys.stderr (where loguru writes) to a file
@contextlib.contextmanager
def redirect_to_file(filepath):
    """Redirects all logger output (stderr) to a specified file."""
    original_stderr = sys.stderr
    try:
        # Open the file and set it as the new stderr
        with open(filepath, 'w', encoding='utf-8') as f:
            sys.stderr = f
            # Reconfigure loguru to use the new stderr file handle
            # Note: Must call setup_logger() to refresh the handlers
            setup_logger() 
            yield
    finally:
        # Restore the original stderr
        sys.stderr = original_stderr
        setup_logger() # Restore loguru to the original stderr (console)

def run_scenario(manager: CoordinatorAgent, name: str, query: str, file: str = None, filename: str = None):
    """Runs a single test scenario and captures the output.
    Accepts either 'file' (existing SCENARIOS keys) or 'filename' (preferred).
    """
    out_name = filename or file
    if out_name is None:
        raise ValueError("Either 'file' or 'filename' must be provided for run_scenario.")
    
    output_path = os.path.join(OUTPUTS_DIR, out_name)
    
    # Use the context manager to redirect output
    with redirect_to_file(output_path):
        # Log the start of the scenario inside the file
        log.info(f"===== SCENARIO START: {name} =====")
        log.info(f"USER INPUT: {query}")
        
        # Execute the query
        final_answer = manager.process_query(query)
        
        # Log the final answer and the end of the scenario
        log.info("FINAL ANSWER GENERATED:")
        log.info(final_answer)
        log.info(f"===== SCENARIO END: {name} =====")

    print(f"[SUCCESS] Scenario '{name}' completed. Output saved to {output_path}")

def main():
    """Main function to initialize and run all test scenarios."""
    
    # 0. Setup Environment
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)
        
    print("--- Starting Multi-Agent System Test Runs ---")
    
    # 1. Initialize the Coordinator (which initializes all other agents/memory)
    try:
        manager = CoordinatorAgent()
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize CoordinatorAgent. Check .env and dependencies. Error: {e}")
        return

    # 2. Run Simple Query
    run_scenario(manager, "Simple Query", **SCENARIOS["Simple Query"])
    
    # 3. Run Complex Query
    run_scenario(manager, "Complex Query", **SCENARIOS["Complex Query"])

    # 4. Run Multi-step Query
    run_scenario(manager, "Multi-step", **SCENARIOS["Multi-step"])
    
    # 5. Run Collaborative Query
    run_scenario(manager, "Collaborative", **SCENARIOS["Collaborative"])

    # 6. Run Memory Test (Crucial two-step process)
    # --- Step 1: Store the information ---
    print("\n--- Running Memory Test (Step 1: Store Knowledge) ---")
    run_scenario(manager, "Memory Test (Step 1: Store)", **SCENARIOS["Memory Test (Step 1: Store)"])
    
    # Give FAISS/disk time to settle (optional, but safe)
    time.sleep(1) 
    
    # --- Step 2: Retrieve the information via similarity search, demonstrating NO redundant research ---
    print("\n--- Running Memory Test (Step 2: Reuse Knowledge) ---")
    run_scenario(manager, "Memory Test (Step 2: Reuse)", **SCENARIOS["Memory Test (Step 2: Reuse)"])
    
    print("\n--- All Test Scenarios Completed Successfully ---")


if __name__ == "__main__":
    main()