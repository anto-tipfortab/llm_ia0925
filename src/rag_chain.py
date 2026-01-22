"""
RAG Chain module - combines vector retrieval with LLM generation.
Supports multi-turn conversation and function calling.
"""

import json
from src.logger import logger

SYSTEM_PROMPT_PATH = "src/system_prompt.txt"


def load_prompt(path: str) -> str:
    """Load prompt from text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RAGChain:
    """Combines vector store retrieval with LLM to generate answers."""
    
    def __init__(self, client, vector_store, weather_service=None, prompt_path: str = SYSTEM_PROMPT_PATH, max_history: int = 5):
        self.client = client
        self.vector_store = vector_store
        self.weather_service = weather_service
        self.system_prompt = load_prompt(prompt_path)
        self.max_history = max_history
        self.history = []
        
        # Build tools list
        self.tools = []
        if weather_service:
            self.tools.append(weather_service.get_tool_schema())
        
        logger.info(f"RAGChain initialized (max_history={max_history}, tools={len(self.tools)})")
    
    def _build_context(self, chunks: list) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get('page', 'N/A')
            context_parts.append(f"[Fuente: página {page}]\n{chunk.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _trim_history(self):
        """Keep only the last max_history turns (user + assistant pairs)."""
        max_messages = self.max_history * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]
            logger.info(f"History trimmed to {len(self.history)} messages")
    
    def _handle_tool_call(self, tool_call) -> str:
        """Execute a tool call and return the result."""
        function_name = tool_call.function.name
        logger.info(f"Tool call requested: {function_name}")
        logger.info(f"Tool call arguments: {tool_call.function.arguments}")
        
        if function_name == "get_weather" and self.weather_service:
            result = self.weather_service.parse_tool_call(tool_call)
            logger.info(f"Tool result: {result}")
            return json.dumps(result, ensure_ascii=False)
        
        logger.warning(f"Unknown function: {function_name}")
        return json.dumps({"error": True, "message": f"Función desconocida: {function_name}"})
    
    def query(self, question: str, k: int = 3) -> dict:
        """Process a question with conversation history, RAG, and function calling."""
        
        # Step 1: Retrieve relevant chunks
        chunks = self.vector_store.search(question, k=k)
        logger.info(f"Retrieved {len(chunks)} chunks for: {question[:50]}...")
        
        # Step 2: Build context
        context = self._build_context(chunks)
        
        # Step 3: Build messages with history
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        
        # Add current question with context
        user_message = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"
        messages.append({"role": "user", "content": user_message})
        
        # Step 4: Get LLM response (with tools if available)
        tool_called = False
        if self.tools:
            response = self.client.get_completion_with_functions(messages, self.tools)
            
            # Check if LLM wants to call a function
            if response["tool_calls"]:
                tool_called = True
                tool_call = response["tool_calls"][0]
                
                # Execute the tool
                tool_result = self._handle_tool_call(tool_call)
                
                # Add assistant's tool call to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
                
                # Get final response from LLM with tool result
                answer = self.client.get_completion(messages)
                logger.info(f"Generated answer after tool call: {len(answer)} chars")
            else:
                answer = response["content"]
                logger.info(f"Generated answer (no tool call): {len(answer)} chars")
        else:
            answer = self.client.get_completion(messages)
            logger.info(f"Generated answer: {len(answer)} chars")
        
        # Step 5: Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        
        # Step 6: Return answer with metadata
        sources = [{"page": c.metadata.get('page', 'N/A'), "preview": c.page_content[:100]} for c in chunks]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "tool_called": tool_called
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> list:
        """Get current conversation history."""
        return self.history.copy()