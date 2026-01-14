"""
Data Loader module
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from .logger import logger


class DataLoader:
    """Handles loading and basic processing of documents."""
    
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        self.pages = []
        logger.info(f"DataLoader initialized with: {file_path}")
    
    def load(self) -> list:
        """Load document and return list of pages."""
        loader = PyPDFLoader(self.file_path)
        self.pages = loader.load()
        logger.info(f"Loaded {len(self.pages)} pages from {self.file_path}")
        return self.pages
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded document."""
        if not self.pages:
            self.load()
        
        total_chars = sum(len(p.page_content) for p in self.pages)
        total_words = sum(len(p.page_content.split()) for p in self.pages)
        
        return {
            "num_pages": len(self.pages),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_page": total_chars // len(self.pages) if self.pages else 0
        }
    
    def get_page(self, index: int) -> str:
        """Get content of a specific page."""
        if not self.pages:
            self.load()
        
        if 0 <= index < len(self.pages):
            return self.pages[index].page_content
        raise IndexError(f"Page index {index} out of range")
    
    def get_all_text(self) -> str:
        """Get all document text concatenated."""
        if not self.pages:
            self.load()
        
        return "\n\n".join(p.page_content for p in self.pages)