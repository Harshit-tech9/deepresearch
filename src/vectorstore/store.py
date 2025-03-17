"""GPU-optimized vector store module with improved error handling."""

import os
import json
import torch
import faiss
import numpy as np
import hashlib
import argparse
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
    """GPU-optimized class to handle vector store operations with improved robustness."""
    
    def __init__(self, 
                 model_name: str = EMBEDDING_MODEL,
                 device: str = GPU_DEVICE,
                 embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
                 index_type: str = FAISS_INDEX_TYPE):
        """
        Initialize the vector store manager with GPU support.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run embeddings on ('cpu' or 'cuda')
            embedding_batch_size: Batch size for embedding processing
            index_type: Type of FAISS index to use ('Flat' or 'IVFFlat')
        """
        self.model_name = model_name
        self.embedding_batch_size = embedding_batch_size
        self.index_type = index_type
        
        # Auto-detect device if GPU is not available
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        self.device = device
        
        # Set up embeddings with optimal settings for the available device
        model_kwargs = {'device': self.device}
        encode_kwargs = {
            'normalize_embeddings': True,  # Normalizing improves search quality
            'batch_size': self.embedding_batch_size
        }
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        self.is_index_trained = False if index_type == "IVFFlat" else True
        
        logger.info(f"Initialized vector store with model: {model_name} on device: {device} using index type: {index_type}")
    
    def _get_faiss_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index with appropriate settings based on availability.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index
        """
        # Choose index type based on setting and available hardware
        if self.index_type == "IVFFlat":
            # For IVFFlat, we'll create the index but we'll need to train it later
            # This requires a sample of vectors
            n_lists = max(4, min(1000, int(np.sqrt(10000))))  # Rule of thumb for clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_lists)
            # Set nprobe - how many clusters to visit during search
            index.nprobe = min(10, n_lists)
            logger.info(f"Created IVFFlat index with {n_lists} lists, nprobe={index.nprobe}")
        else:
            # Simple flat index - always works but slower for large datasets
            logger.info("Created Flat index - exact but slower search")
            index = faiss.IndexFlatL2(dimension)
            
        # Convert to GPU index if available and requested
        if USE_GPU and self.device == 'cuda':
            try:
                res = faiss.StandardGpuResources()
                # Use temp memory fraction to control GPU memory usage
                res.setTempMemoryFraction(0.3)  # Use 30% of GPU memory for temp allocations
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU-accelerated FAISS index")
                return gpu_index
            except Exception as e:
                logger.warning(f"Failed to create GPU index, falling back to CPU: {str(e)}")
        
        return index
    
    def _initialize_vector_store(self):
        """
        Initialize FAISS vector store with robust handling.
        
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
            # Create a simpler fallback index if the requested one fails
            try:
                logger.warning("Falling back to simple Flat index")
                self.index_type = "Flat"  # Override to simpler index
                index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                return FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
            except Exception as e2:
                logger.error(f"Critical error creating fallback index: {str(e2)}")
                raise
    
    def _train_index_if_needed(self, sample_data: List[Document], batch_size: int = 64) -> bool:
        """
        Train the index if it's an IVFFlat index and not yet trained.
        
        Args:
            sample_data: Sample documents to generate training vectors
            batch_size: Batch size for processing
            
        Returns:
            True if training was successful or not needed, False otherwise
        """
        if self.is_index_trained or self.index_type != "IVFFlat":
            return True
            
        try:
            logger.info("Training IVFFlat index with sample data...")
            
            # Generate embeddings for sample data
            sample_texts = [doc.page_content for doc in sample_data[:min(len(sample_data), 1000)]]
            embeddings = self.embeddings.embed_documents(sample_texts)
            
            # Convert to numpy array
            training_vectors = np.array(embeddings).astype(np.float32)
            
            # Get the raw index
            if hasattr(self.vector_store, 'index'):
                raw_index = self.vector_store.index
                
                # If it's a GPU index, we need to get the CPU version for training
                if isinstance(raw_index, faiss.GpuIndex):
                    logger.info("Converting GPU index to CPU for training")
                    cpu_index = faiss.index_gpu_to_cpu(raw_index)
                    
                    # Train the CPU index
                    logger.info(f"Training index with {len(training_vectors)} vectors")
                    cpu_index.train(training_vectors)
                    
                    # Convert back to GPU
                    res = faiss.StandardGpuResources()
                    res.setTempMemoryFraction(0.3)
                    self.vector_store.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                else:
                    # Train the CPU index directly
                    logger.info(f"Training index with {len(training_vectors)} vectors")
                    raw_index.train(training_vectors)
                
                self.is_index_trained = True
                logger.info("Index training completed successfully")
                return True
            else:
                logger.error("Vector store has no index attribute")
                return False
        except Exception as e:
            logger.error(f"Error training index: {str(e)}")
            # Fall back to Flat index if training fails
            try:
                logger.warning("Training failed, falling back to Flat index")
                self.index_type = "Flat"
                index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                
                # If using GPU, convert to GPU index
                if USE_GPU and self.device == 'cuda':
                    try:
                        res = faiss.StandardGpuResources()
                        res.setTempMemoryFraction(0.3)
                        index = faiss.index_cpu_to_gpu(res, 0, index)
                    except Exception as e2:
                        logger.warning(f"Failed to convert fallback index to GPU: {str(e2)}")
                
                # Create new vector store with flat index
                self.vector_store = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                self.is_index_trained = True  # Flat index doesn't need training
                return True
            except Exception as e2:
                logger.error(f"Critical error creating fallback index: {str(e2)}")
                return False
    
    @track_performance
    def add_documents(self, documents: List[Document], batch_size: int = 64) -> int:
        """
        Add documents to vector store in batches, with training if needed.
        
        Args:
            documents: List of documents to add
            batch_size: Size of batches for processing
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return 0
        
        try:
            # First train the index if needed using a sample of the documents
            if not self._train_index_if_needed(documents, batch_size):
                logger.error("Index training failed, cannot add documents")
                return 0
            
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
            # Try with smaller batch if possible
            if batch_size > 8 and len(documents) > 8:
                try:
                    logger.warning(f"Retrying with smaller batch size (8)...")
                    return self.add_documents(documents, batch_size=8)
                except Exception as e2:
                    logger.error(f"Failed with smaller batch size too: {str(e2)}")
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
            
            # Check if vector store has documents
            if len(self.vector_store.docstore._dict) == 0:
                logger.warning("Vector store is empty - no documents to search")
                return []
            
            # Use MMR for diversity in results
            results = self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=fetch_k
            )
            
            logger.info(f"Found {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            # Try simple similarity search as fallback
            try:
                logger.warning("Falling back to simple similarity search...")
                results = self.vector_store.similarity_search(query, k=k)
                return results
            except Exception as e2:
                logger.error(f"Fallback search also failed: {str(e2)}")
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
                "index_type": self.index_type,
                "is_index_trained": self.is_index_trained,
                "documents_count": len(self.vector_store.docstore._dict)
            }
            
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Vector store saved to {os.path.abspath(path)} with {metadata['documents_count']} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self, path: str = "vector_store") -> bool:
        """
        Load vector store from disk with error handling.
        
        Args:
            path: Path to load vector store from
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Vector store path {path} does not exist")
                return False
            
            # Load metadata if available
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    logger.info(f"Loading vector store with metadata: {metadata}")
                    
                    # Update instance variables from metadata
                    if "index_type" in metadata:
                        self.index_type = metadata["index_type"]
                    if "is_index_trained" in metadata:
                        self.is_index_trained = metadata["is_index_trained"]
                except Exception as e:
                    logger.warning(f"Error reading metadata: {str(e)}")
            
            # Load vector store
            self.vector_store = FAISS.load_local(
                path, self.embeddings, allow_dangerous_deserialization=True
            )
            
            # If using GPU, convert index to GPU
            if USE_GPU and self.device == 'cuda' and hasattr(self.vector_store, "index"):
                try:
                    res = faiss.StandardGpuResources()
                    res.setTempMemoryFraction(0.3)
                    self.vector_store.index = faiss.index_cpu_to_gpu(
                        res, 0, self.vector_store.index
                    )
                    logger.info("Converted loaded index to GPU index")
                except Exception as e:
                    logger.warning(f"Failed to convert loaded index to GPU: {str(e)}")
            
            logger.info(f"Vector store loaded from {os.path.abspath(path)} with {len(self.vector_store.docstore._dict)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False

    # Chunking methods for document preprocessing
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
        if not documents:
            return []
            
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

def get_safe_path(query, base_dir="vector_stores"):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a vector store")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--output-base", default="content", help="Base directory for content")
    parser.add_argument("--vector-base", default="vector_stores", help="Base directory for vector stores")
    parser.add_argument("--use-ivf", action="store_false", dest="force_flat", help="Allow IVFFlat index")
    
    args = parser.parse_args()
    
    manager, out_dir, vec_dir = initialize_vector_store(
        query=args.query,
        output_base=args.output_base,
        vector_base=args.vector_base,
        force_flat=args.force_flat
    )
    
    print(f"Initialized vector store:")
    print(f"Output directory: {out_dir}")
    print(f"Vector store directory: {vec_dir}")
    print(f"Index type: {manager.index_type}")