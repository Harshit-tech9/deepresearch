"""Vector store management functionality."""

import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from .store import VectorStoreManager
from ..utils.logging import get_logger

logger = get_logger(__name__)

def get_safe_path(query: str, base_dir: str = "vector_stores") -> str:
    """Generate a safe directory name from a query string."""
    # Create a hash of the query
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    
    # Create a safe name from the query
    safe_name = "".join(c if c.isalnum() else "_" for c in query.lower())
    safe_name = safe_name[:30]  # Limit length
    
    # Combine with hash for uniqueness
    path = f"{safe_name}_{query_hash}"
    
    return os.path.join(base_dir, path)

def initialize_vector_store(
    query: str,
    output_base: str = "content",
    vector_base: str = "vector_stores",
    force_flat: bool = True
) -> Tuple[VectorStoreManager, str, str]:
    """
    Initialize a vector store for a query.
    
    Args:
        query: The search query
        output_base: Base directory for content
        vector_base: Base directory for vector stores
        force_flat: Force using Flat index type
        
    Returns:
        Tuple of (VectorStoreManager, output_dir, vector_dir)
    """
    # Create safe directory names
    output_dir = get_safe_path(query, output_base)
    vector_dir = get_safe_path(query, vector_base)
    
    # Ensure base directories exist
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(vector_base, exist_ok=True)
    
    # Initialize vector store manager
    vector_store_manager = VectorStoreManager(index_type="Flat" if force_flat else "IVFFlat")
    
    return vector_store_manager, output_dir, vector_dir

def create_vector_store_summary(
    query: str,
    output_dir: str,
    vector_dir: str,
    vector_store_manager: Optional[VectorStoreManager] = None,
    extraction_results: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a summary of vector store creation.
    
    Args:
        query: The search query
        output_dir: Output directory path
        vector_dir: Vector store directory path
        vector_store_manager: Optional VectorStoreManager instance
        extraction_results: Optional extraction results
        
    Returns:
        Summary dictionary
    """
    doc_count = 0
    if vector_store_manager and hasattr(vector_store_manager, 'vector_store'):
        doc_count = len(vector_store_manager.vector_store.docstore._dict)
    
    urls_processed = 0
    successful_extractions = 0
    failed_extractions = 0
    
    if extraction_results:
        urls_processed = extraction_results.get('total_urls', 0)
        successful_extractions = extraction_results.get('successful_count', 0)
        failed_extractions = extraction_results.get('failed_count', 0)
    
    summary = {
        "query": query,
        "output_dir": output_dir,
        "vector_dir": vector_dir,
        "created_at": datetime.now().isoformat(),
        "urls_processed": urls_processed,
        "successful_extractions": successful_extractions,
        "failed_extractions": failed_extractions,
        "documents_count": doc_count,
        "has_vector_store": vector_store_manager is not None
    }
    
    return summary 