"""Configuration settings for the application."""

import os
import torch

# System settings
DEFAULT_OUTPUT_DIR = "extracted_content"
DEFAULT_VECTOR_STORE_PATH = "vector_store"
MAX_CONCURRENT_REQUESTS = 15

# GPU settings
USE_GPU = torch.cuda.is_available()
GPU_DEVICE = "cuda" if USE_GPU else "cpu"
GPU_MEMORY_FRACTION = 0.8  # Use 80% of available GPU memory

# If we have multiple GPUs, specify which one to use
GPU_ID = 0
if USE_GPU and torch.cuda.device_count() > 1:
    torch.cuda.set_device(GPU_ID)

# Get available GPU memory
GPU_MEMORY = None
if USE_GPU:
    try:
        # Get total memory in bytes and convert to GB
        total_memory = torch.cuda.get_device_properties(GPU_ID).total_memory
        GPU_MEMORY = total_memory / (1024 ** 3)  # Convert to GB
    except Exception as e:
        print(f"Failed to get GPU memory: {e}")
        GPU_MEMORY = 4  # Default to 4GB if unable to detect

# Vector store settings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768  # Dimension for the embedding model
EMBEDDING_BATCH_SIZE = 32  # Process embeddings in batches
FAISS_INDEX_TYPE = "Flat"  # Use IVFFlat for larger datasets with GPU

# LLM settings
DEFAULT_LLM_MODEL = "llama-3.3-70b-specdec"
DEFAULT_LLM_TEMPERATURE = 0.1
DEFAULT_API_KEY = "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"

# Search settings
MAX_SEARCH_RESULTS = 40
DEFAULT_RETRIEVAL_RESULTS = 5

# Chunk settings
DEFAULT_CHUNK_SIZE = 3000