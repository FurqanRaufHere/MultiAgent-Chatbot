#!/usr/bin/env python3
"""Interactive chat REPL using the CoordinatorAgent.

Run:
    python scripts/chat.py

Type `exit` or `quit` to leave.
"""
import os
import sys
import pathlib
from dotenv import load_dotenv

# Ensure project root is on sys.path so local packages (agents, utils, memory) import correctly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv()

from agents.coordinator import CoordinatorAgent
from utils.logger import setup_logger, log


def main():
    setup_logger()
    log.info("Starting interactive chat REPL...")

    manager = CoordinatorAgent()

    print("Interactive chat started. Type 'quit' or 'exit' to stop.")
    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                break

            log.info("User prompt received (interactive)")
            response = manager.process_query(user_input)

            # Print and log the response
            print("Agent:", response)
            log.info("Interactive response: {resp}", resp=response)

    except KeyboardInterrupt:
        print()
        log.info("Chat REPL interrupted by user.")

    log.info("Exiting chat REPL.")


if __name__ == '__main__':
    main()
