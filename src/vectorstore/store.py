"""GPU-optimized vector store module."""

import os
import json
import torch
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ..utils.logging import get_logger
from ..utils.metrics import track_performance
from config.settings import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDING_BATCH_SIZE,
    GPU_DEVICE, USE_GPU, FAISS_INDEX_TYPE
)

logger = get_logger(__name__)

class VectorStoreManager:
    """GPU-optimized class to handle vector store operations."""
    
    def __init__(self, 
                 model_name: str = EMBEDDING_MODEL,
                 device: str = GPU_DEVICE,
                 embedding_batch_size: int = EMBEDDING_BATCH_SIZE):
        """
        Initialize the vector store manager with GPU support.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run embeddings on ('cpu' or 'cuda')
            embedding_batch_size: Batch size for embedding processing
        """
        self.model_name = model_name
        self.device = device
        self.embedding_batch_size = embedding_batch_size
        
        # Set up embeddings with GPU support if available
        model_kwargs = {'device': self.device}
        encode_kwargs = {
            'normalize_embeddings': True,  # Normalizing can improve search quality
            'batch_size': self.embedding_batch_size  # Process in batches
        }
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        logger.info(f"Initialized vector store with model: {model_name} on device: {device}")
    
    def _get_faiss_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index with GPU support if available.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index
        """
        # Choose index type based on dataset size and available GPU
        if FAISS_INDEX_TYPE == "Flat":
            # Simple flat index - exact search, slower but more accurate
            index = faiss.IndexFlatL2(dimension)
        elif FAISS_INDEX_TYPE == "IVFFlat":
            # IVF index - faster search with slight accuracy tradeoff
            n_lists = max(4, min(1000, int(np.sqrt(10000))))  # Rule of thumb for number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
            index.nprobe = min(10, n_lists)  # Number of clusters to visit during search
        else:
            # Default to flat index
            index = faiss.IndexFlatL2(dimension)
        
        # Convert to GPU index if available
        if USE_GPU:
            try:
                res = faiss.StandardGpuResources()
                # Use temp memory fraction to control GPU memory usage
                res.setTempMemoryFraction(0.3)  # Use 30% of GPU memory for temporary allocations
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU-accelerated FAISS index")
                return gpu_index
            except Exception as e:
                logger.warning(f"Failed to create GPU index, falling back to CPU: {str(e)}")
        
        return index
    
    def _initialize_vector_store(self):
        """
        Initialize FAISS vector store with GPU support.
        
        Returns:
            FAISS vector store
        """
        try:
            # Get embedding dimension
            dimension = EMBEDDING_DIMENSION
            
            # Create FAISS index
            index = self._get_faiss_index(dimension)
            
            # Create vector store
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    @track_performance
    def add_documents(self, documents: List[Document], batch_size: int = 64) -> int:
        """
        Add documents to vector store in batches.
        
        Args:
            documents: List of documents to add
            batch_size: Size of batches for processing
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        try:
            added_count = 0
            logger.info(f"Adding {len(documents)} documents to vector store in batches of {batch_size}")
            
            # Process in batches to manage memory
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                # Generate UUIDs
                uuids = [str(uuid4()) for _ in range(len(batch))]
                
                # Add batch to vector store
                self.vector_store.add_documents(documents=batch, ids=uuids)
                
                added_count += len(batch)
                logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
            
            logger.info(f"Successfully added {added_count} documents to vector store")
            return added_count
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return 0
    
    @track_performance
    def search(self, query: str, k: int = 5, fetch_k: int = None) -> List[Document]:
        """
        Search vector store for similar documents with optimized retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of documents to fetch before MMR reranking
            
        Returns:
            List of similar documents
        """
        try:
            if fetch_k is None:
                fetch_k = min(4 * k, 100)  # Fetch more docs than needed for reranking
            
            # Use MMR for diversity in results
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
            
            logger.info(f"Found {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def save(self, path: str = "vector_store") -> bool:
        """
        Save vector store to disk with metadata.
        
        Args:
            path: Path to save vector store
            
        Returns:
            Success status
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save the vector store
            self.vector_store.save_local(path)
            
            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_batch_size": self.embedding_batch_size,
                "saved_at": datetime.now().isoformat(),
                "index_type": FAISS_INDEX_TYPE,
                "documents_count": len(self.vector_store.docstore._dict)
            }
            
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {os.path.abspath(path)}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self, path: str = "vector_store") -> bool:
        """
        Load vector store from disk.
        
        Args:
            path: Path to load vector store from
            
        Returns:
            Success status
        """
        try:
            if os.path.exists(path):
                # Load metadata if available
                metadata_path = os.path.join(path, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loading vector store with metadata: {metadata}")
                
                # Load vector store
                self.vector_store = FAISS.load_local(
                    path, self.embeddings, allow_dangerous_deserialization=True
                )
                
                # If using GPU, convert index to GPU
                if USE_GPU and hasattr(self.vector_store, "index"):
                    try:
                        res = faiss.StandardGpuResources()
                        res.setTempMemoryFraction(0.3)
                        self.vector_store.index = faiss.index_cpu_to_gpu(
                            res, 0, self.vector_store.index
                        )
                        logger.info("Converted loaded index to GPU index")
                    except Exception as e:
                        logger.warning(f"Failed to convert loaded index to GPU: {str(e)}")
                
                logger.info(f"Vector store loaded from {os.path.abspath(path)}")
                return True
            else:
                logger.warning(f"Vector store path {path} does not exist")
                return False
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    # Add chunking methods for document preprocessing
    @staticmethod
    def chunk_documents(documents: List[Document], chunk_size: int = 1000, 
                       chunk_overlap: int = 100) -> List[Document]:
        """
        Split documents into smaller chunks for better processing.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List of chunked documents
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.page_content
            
            # Skip if text is too short
            if len(text) <= chunk_size:
                chunked_docs.append(doc)
                continue
            
            # Split into chunks
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 100:  # Skip very small chunks
                    continue
                chunks.append(chunk)
            
            # Create new document for each chunk
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk": i,
                        "chunk_of": len(chunks)
                    }
                )
                chunked_docs.append(new_doc)
        
        return chunked_docs