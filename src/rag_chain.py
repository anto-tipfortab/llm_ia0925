"""
RAG Chain module - combines vector retrieval with LLM generation.
Supports multi-turn conversation with history management.
"""

from src.logger import logger

SYSTEM_PROMPT_PATH = "src/system_prompt.txt"


def load_prompt(path: str) -> str:
    """Load prompt from text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RAGChain:
    """Combines vector store retrieval with LLM to generate answers."""
    
    def __init__(self, client, vector_store, prompt_path: str = SYSTEM_PROMPT_PATH, max_history: int = 5):
        self.client = client
        self.vector_store = vector_store
        self.system_prompt = load_prompt(prompt_path)
        self.max_history = max_history  # Max conversation turns to keep
        self.history = []  # Conversation history
        
        logger.info(f"RAGChain initialized (max_history={max_history})")
    
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
    
    def query(self, question: str, k: int = 3) -> dict:
        """Process a question with conversation history and return answer with sources."""
        
        # Step 1: Retrieve relevant chunks
        chunks = self.vector_store.search(question, k=k)
        logger.info(f"Retrieved {len(chunks)} chunks for: {question[:50]}...")
        
        # Step 2: Build context
        context = self._build_context(chunks)
        
        # Step 3: Build messages with history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        messages.extend(self.history)
        
        # Add current question with context
        user_message = f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"
        messages.append({"role": "user", "content": user_message})
        
        # Step 4: Get LLM response
        answer = self.client.get_completion(messages)
        logger.info(f"Generated answer of {len(answer)} chars")
        
        # Step 5: Update history (store without context to save tokens)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        
        # Step 6: Return answer with metadata
        sources = [{"page": c.metadata.get('page', 'N/A'), "preview": c.page_content[:100]} for c in chunks]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> list:
        """Get current conversation history."""
        return self.history.copy()