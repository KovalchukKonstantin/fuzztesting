"""
LLM Client service for handling interactions with Language Models.
"""
import os
import json
import re
import logging
from typing import Optional, Any, Dict, List
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence noisy httpx logs (used by OpenAI SDK)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Silence uvicorn access logs (every HTTP request)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def extract_json_from_response(content: str) -> str:
    """
    Extract JSON from LLM response, handling markdown code blocks and prose.
    
    Tries multiple strategies:
    1. Extract from ```json ... ``` code block
    2. Extract from ``` ... ``` code block
    3. Find JSON array/object directly in text
    4. Return stripped content as fallback
    """
    # Strategy 1: Look for ```json ... ``` block
    json_block_match = re.search(r'```json\s*([\s\S]*?)```', content)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    # Strategy 2: Look for ``` ... ``` block (any language)
    code_block_match = re.search(r'```\s*([\s\S]*?)```', content)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Strategy 3: Try to find a JSON array or object directly
    # Look for content starting with [ and ending with ]
    array_match = re.search(r'(\[[\s\S]*\])', content)
    if array_match:
        return array_match.group(1).strip()
    
    # Look for content starting with { and ending with }
    object_match = re.search(r'(\{[\s\S]*\})', content)
    if object_match:
        return object_match.group(1).strip()
    
    # Fallback: return stripped content
    return content.strip()


class LLMClient:
    """
    Client for interacting with LLMs (defaulting to OpenAI).
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to env var OPENAI_API_KEY)
            model: Model identifier
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found. LLM calls will fail unless mocked.")
            
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        self.model = model
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.7
    ) -> str:
        """
        Complete a prompt using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            json_mode: Whether to enforce JSON output
            temperature: Sampling temperature
            
        Returns:
            The generated text response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response_format = {"type": "json_object"} if json_mode else None
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            return content
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True
    )
    async def complete_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        validator: Optional[callable] = None
    ) -> Any:
        """
        Get structured JSON response.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            validator: Optional validation function that raises Exception on invalid data
            
        Returns:
            Parsed JSON object (dict or list)
        """
        # We don't force json_mode=True here because it enforces {"type": "json_object"}
        # which disallows top-level lists, which our templates use.
        content = await self.complete(prompt, system_prompt, json_mode=False)
        
        # Extract JSON from response (handles markdown code blocks and prose)
        cleaned_content = extract_json_from_response(content)
        
        try:
            data = json.loads(cleaned_content)
            
            # Run custom validation if provided
            if validator:
                try:
                    validator(data)
                except Exception as e:
                    logger.warning(f"Validation failed: {e}. Retrying...")
                    raise ValueError(f"Validation failed: {e}")
            
            return data
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {content}")
            raise ValueError("LLM did not return valid JSON")
