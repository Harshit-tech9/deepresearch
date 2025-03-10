"# deepresearch" 
# GPU-Accelerated Research and RAG System

This project provides a modular, GPU-accelerated system for research, document retrieval, and RAG (Retrieval Augmented Generation) operations.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU
- Required libraries (see Installation)

## Installation

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install langchain langchain_huggingface langchain_groq langchain_community faiss-gpu duckduckgo_search bs4 nest_asyncio psutil torch
```

## GPU Configuration

This system is optimized for GPUs with at least 4GB of memory. It includes:

- Automatic detection of GPU availability
- Memory management to prevent OOM errors
- Batch processing for efficient GPU usage
- FAISS GPU acceleration for vector operations

## Verifying GPU Setup

To verify your GPU is properly detected and configured:

```bash
python check_gpu.py
```

This will run diagnostic tests to ensure your GPU is available and working with PyTorch and FAISS.

## Directory Structure

```
project_root/
├── config/
│   └── settings.py            # Configuration settings
├── src/
│   ├── scraping/
│   │   └── scraper.py         # Web scraping functionality
│   ├── search/
│   │   └── searcher.py        # Search capabilities
│   ├── vectorstore/
│   │   └── store.py           # Vector store operations
│   ├── llm/
│   │   └── processor.py       # LLM processing and RAG
│   └── utils/
│       ├── logging.py         # Logging utilities
│       └── metrics.py         # Performance metrics
├── examples/
│   └── example_workflows.py   # Example usage scenarios
├── main.py                    # Main entry point
└── check_gpu.py               # GPU verification utility
```

## Usage Example

```python
from main import enhanced_search_and_vectorize, perform_rag_search

# Run the full pipeline
result = enhanced_search_and_vectorize(
    user_query="future of renewable energy technologies 2030",
    output_dir="renewable_energy_content",
    vector_store_path="renewable_energy_vector_store",
    max_results=20,
    max_concurrent=10
)

# Perform RAG-based search
rag_result = perform_rag_search(
    query="What are the most promising solar storage technologies?",
    vector_store_path="renewable_energy_vector_store",
    k=5
)

# Display the answer
print(rag_result['answer'])
```

## Key Features

1. **GPU-Accelerated Vector Operations**:
   - FAISS GPU indices for faster similarity search
   - Optimized memory usage for large document collections
   - Batch processing for embeddings

2. **Asynchronous Web Scraping**:
   - Concurrent fetching of web content
   - Robust error handling and retries
   - Content extraction and cleaning

3. **Enhanced RAG Capabilities**:
   - Query refinement with LLM
   - Contextual retrieval with MMR reranking
   - Well-structured responses

4. **Performance Monitoring**:
   - Tracking of CPU, RAM, and GPU usage
   - Detailed timing information
   - Benchmarking tools

## Troubleshooting GPU Issues

If you encounter GPU-related issues:

1. Ensure your NVIDIA drivers are up to date
2. Check that PyTorch is installed with CUDA support
3. Monitor GPU memory usage during operations
4. Try reducing batch sizes if you experience OOM errors

## Performance Tips

To maximize performance:

1. Adjust `max_concurrent` based on your internet connection
2. Set appropriate batch sizes for your GPU memory
3. Use chunking for very large documents
4. Enable caching for repeated operations