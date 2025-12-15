import os
import json
from groq import Groq, GroqError
from dotenv import load_dotenv
from utils.logger import log
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

class LLMConnector:
    """Handles communication with the Groq LLM for planning and analysis."""
    
    def __init__(self):
        self.log = log.bind(name="LLMConnector")
        api_key = os.getenv("GROQ_API_KEY")
        
        # Fallback mechanism: Check for API key presence and initialize client
        if not api_key:
            self.log.error("GROQ_API_KEY not found. LLM functionality is DISABLED.")
            self.client = None
            self.model = None
        else:
            try:
                self.client = Groq(api_key=api_key)
                # Using a fast and capable model for orchestration
                self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") 
                self.log.info("Groq LLM connector initialized with model: {model}", model=self.model)
            except Exception as e:
                self.log.error("Failed to initialize Groq client: {error}", error=str(e))
                self.client = None
                self.model = None

    def is_functional(self):
        """Checks if the LLM is available and initialized."""
        return self.client is not None

    def generate_response(self, prompt: str, system_prompt: str, json_mode: bool = False) -> Optional[Any]:
        """Sends a request to the Groq API."""
        if not self.is_functional():
            # Graceful Degradation: Returning None triggers rule-based fallback in calling agents
            return None 

        try:
            response_format = {"type": "json_object"} if json_mode else {"type": "text"}
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format
            )
            
            content = completion.choices[0].message.content
            
            if json_mode:
                # Attempt to parse JSON output
                return json.loads(content)
            
            return content
            
        except GroqError as e:
            self.log.error("Groq API Error: {error}", error=str(e))
            return None
        except json.JSONDecodeError:
            self.log.error("LLM returned malformed JSON. Content: {content}", content=content[:50])
            return None
        except Exception as e:
            self.log.error("Unexpected error during LLM generation: {error}", error=str(e))
            return None