"""
RAG Chain module - combines vector retrieval with LLM generation.
"""

from src.logger import logger

SYSTEM_PROMPT_PATH = "src/system_prompt.txt"


def load_prompt(path: str) -> str:
    """Load prompt from text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class RAGChain:
    """Combines vector store retrieval with LLM to generate answers."""
    
    def __init__(self, client, vector_store, prompt_path: str = SYSTEM_PROMPT_PATH):
        self.client = client
        self.vector_store = vector_store
        self.system_prompt = load_prompt(prompt_path)
        
        logger.info(f"RAGChain initialized with prompt from {prompt_path}")
    
    def _build_context(self, chunks: list) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            page = chunk.metadata.get('page', 'N/A')
            context_parts.append(f"[Fuente: página {page}]\n{chunk.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def query(self, question: str, k: int = 3) -> dict:
        """Process a question and return answer with sources."""
        
        # Step 1: Retrieve relevant chunks
        chunks = self.vector_store.search(question, k=k)
        logger.info(f"Retrieved {len(chunks)} chunks for: {question[:50]}...")
        
        # Step 2: Build context
        context = self._build_context(chunks)
        
        # Step 3: Create messages for LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"}
        ]
        
        # Step 4: Get LLM response
        answer = self.client.get_completion(messages)
        logger.info(f"Generated answer of {len(answer)} chars")
        
        # Step 5: Return answer with metadata
        sources = [{"page": c.metadata.get('page', 'N/A'), "preview": c.page_content[:100]} for c in chunks]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }