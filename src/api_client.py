"""
API Client module
"""

from openai import OpenAI
from .conf import OPENAI_API_KEY, MODEL_CONFIG
from .logger import logger


class OpenAIClient:
    """Wrapper class for OpenAI API interactions."""
    
    def __init__(self, api_key: str = OPENAI_API_KEY, config: dict = MODEL_CONFIG):
        self.client = OpenAI(api_key=api_key)
        self.config = config
        logger.info(f"OpenAI client initialized with model: {config['model']}")
    
    def test_connection(self) -> bool:
        """Test if API connection works."""
        try:
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=10
            )
            logger.info("OpenAI connection test successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
    
    def get_completion(self, messages: list, **kwargs) -> str:
        """Get a chat completion from OpenAI."""
        params = {**self.config, **kwargs}
        
        response = self.client.chat.completions.create(
            messages=messages,
            **params
        )
        
        logger.info(f"Completion received - tokens used: {response.usage.total_tokens}")
        return response.choices[0].message.content
    
    def get_completion_with_functions(self, messages: list, tools: list, **kwargs) -> dict:
        """Get a completion that may include function calls."""
        params = {**self.config, **kwargs}
        
        response = self.client.chat.completions.create(
            messages=messages,
            tools=tools,
            **params
        )
        
        message = response.choices[0].message
        logger.info(f"Completion with tools - finish_reason: {response.choices[0].finish_reason}")
        
        return {
            "content": message.content,
            "tool_calls": message.tool_calls,
            "finish_reason": response.choices[0].finish_reason
        }