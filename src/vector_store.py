"""
Vector Store module - handles chunking, embeddings, and similarity search.
"""

import shutil
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.conf import OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import logger


class VectorStore:
    """Handles document chunking, embedding, storage and retrieval."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = None
        self.chunks = []
        
        # Text splitter config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        logger.info(f"VectorStore initialized (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    def _clear_existing(self):
        """Remove existing vector store to start fresh."""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            logger.info(f"Cleared existing vector store at {self.persist_directory}")
    
    def build_from_documents(self, pages: list, clear_existing: bool = True) -> int:
        """Chunk documents, create embeddings, and store in ChromaDB."""
        
        # Clear old data to avoid duplicates
        if clear_existing:
            self._clear_existing()
        
        # Split pages into chunks
        self.chunks = self.splitter.split_documents(pages)
        logger.info(f"Created {len(self.chunks)} chunks from {len(pages)} pages")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        logger.info(f"Vector store built and persisted to {self.persist_directory}")
        return len(self.chunks)
    
    def load_existing(self) -> bool:
        """Load an existing vector store from disk."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            logger.info(f"Loaded existing vector store from {self.persist_directory}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def search(self, query: str, k: int = 3) -> list:
        """Search for most relevant chunks."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_from_documents() or load_existing() first.")
        
        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Search query: '{query[:50]}...' returned {len(results)} results")
        return results
    
    def search_with_scores(self, query: str, k: int = 3) -> list:
        """Search with relevance scores (lower = more similar)."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"Search with scores: '{query[:50]}...' returned {len(results)} results")
        return results
    
    def get_chunk_stats(self) -> dict:
        """Get statistics about the chunks."""
        if not self.chunks:
            return {"num_chunks": 0}
        
        lengths = [len(c.page_content) for c in self.chunks]
        return {
            "num_chunks": len(self.chunks),
            "avg_chunk_size": sum(lengths) // len(lengths),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths)
        }