"""Main entry point for the application with GPU-optimized pipeline."""

import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

import torch
import nest_asyncio

from config.settings import (
    USE_GPU, GPU_DEVICE, DEFAULT_OUTPUT_DIR, DEFAULT_VECTOR_STORE_PATH,
    MAX_CONCURRENT_REQUESTS, MAX_SEARCH_RESULTS, DEFAULT_RETRIEVAL_RESULTS,
    DEFAULT_LLM_MODEL, DEFAULT_API_KEY
)
from src.utils.logging import get_logger, setup_logger
from src.utils.metrics import performance_monitor, track_performance_context
from src.scraping.improved_scraper import EnhancedWebScraper
from src.search.searcher import DuckDuckGoSearcher
from src.vectorstore.store import VectorStoreManager
from src.llm.processor import QueryRefiner, RAGQueryProcessor

# Enable nested asyncio for Jupyter/IPython environments
nest_asyncio.apply()

# Set up logger
logger = setup_logger('main', log_file=f'logs/main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

def display_gpu_info():
    """Display GPU information."""
    if USE_GPU:
        try:
            logger.info(f"GPU is available: {torch.cuda.is_available()}")
            logger.info(f"GPU device count: {torch.cuda.device_count()}")
            logger.info(f"Current device: {torch.cuda.current_device()}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Get memory info
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)  # GB
            mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)  # GB
            logger.info(f"Memory allocated: {mem_allocated:.2f} GB")
            logger.info(f"Memory reserved: {mem_reserved:.2f} GB")
            
            return True
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {str(e)}")
            return False
    else:
        logger.info("GPU is not available or disabled. Using CPU.")
        return False

# Remove decorator and use context manager inside
async def enhanced_search_and_vectorize_async(
    user_query: str,
    refine_query: bool = True,
    max_results: int = MAX_SEARCH_RESULTS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    save_vector_store: bool = True,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Run a complete search, extraction, and vectorization pipeline with GPU acceleration.
    
    Args:
        user_query: Original user search query
        refine_query: Whether to refine the query using LLM
        max_results: Maximum number of search results
        output_dir: Directory to save extracted content
        max_concurrent: Maximum concurrent requests
        vector_store_path: Path to save vector store
        llm_model: Name of the LLM for query refinement
        save_vector_store: Whether to save the vector store to disk
        api_key: API key for the LLM service
        use_cache: Whether to use caching
        
    Returns:
        Results dictionary
    """
    with track_performance_context("enhanced_search_and_vectorize"):
        logger.info(f"Starting enhanced search for query: '{user_query}'")
        start_time = time.time()
        
        # Step 1: Refine query if enabled
        search_query = user_query
        refined = False
        
        if refine_query:
            try:
                refiner = QueryRefiner(model_name=llm_model, api_key=api_key)
                refined_query = refiner.rewrite_query(user_query, use_cache=use_cache)
                search_query = refined_query
                refined = True
                logger.info(f"Original query: {user_query}")
                logger.info(f"Refined query: {refined_query}")
            except Exception as e:
                logger.error(f"Query refinement failed: {str(e)}")
                logger.info("Proceeding with original query.")
        
        # Step 2: Search DuckDuckGo
        searcher = DuckDuckGoSearcher()
        urls = searcher.search(search_query, max_results=max_results, use_cache=use_cache)
        
        logger.info(f"Found {len(urls)} URLs for query: '{search_query}'")
        
        # Step 3: Extract content
        logger.info(f"Starting extraction of {len(urls)} URLs...")
        
        # Initialize scraper
        scraper = EnhancedWebScraper(output_dir, max_concurrent=max_concurrent)
        
        # Process URLs
        results = await scraper.process_urls(urls)
        
        # Step 4: Initialize vector store with GPU support
        logger.info("Initializing vector store with GPU support...")
        vector_store_manager = VectorStoreManager()
        
        # Step 5: Chunk documents for better embedding
        documents = results['documents']
        chunked_docs = vector_store_manager.chunk_documents(documents)
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        
        # Step 6: Add documents to vector store
        doc_count = vector_store_manager.add_documents(chunked_docs)
        logger.info(f"Added {doc_count} document chunks to vector store")
        
        # Step 7: Save vector store
        if save_vector_store and doc_count > 0:
            vector_store_manager.save(vector_store_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Save query information
        query_info = {
            "original_query": user_query,
            "refined_query": search_query if refined else None,
            "query_refinement_used": refined,
            "timestamp": datetime.now().isoformat(),
            "processing_time": processing_time
        }
        
        with open(os.path.join(output_dir, 'query_info.json'), 'w', encoding='utf-8') as f:
            import json
            json.dump(query_info, f, indent=2)
        
        logger.info("\nEnhanced Search and Vectorization Summary:")
        logger.info(f"Original query: {user_query}")
        if refined:
            logger.info(f"Refined query: {search_query}")
        logger.info(f"Total URLs processed: {results['total_urls']}")
        logger.info(f"Successful extractions: {results['successful_count']}")
        logger.info(f"Failed extractions: {results['failed_count']}")
        logger.info(f"Documents added to vector store: {doc_count}")
        logger.info(f"Total time: {processing_time:.2f} seconds")
        logger.info(f"Results saved to: {os.path.abspath(output_dir)}")
        if save_vector_store and doc_count > 0:
            logger.info(f"Vector store saved to: {os.path.abspath(vector_store_path)}")
        
        # Return the results and vector store manager
        return {
            "query_info": query_info,
            "extraction_results": results,
            "vector_store_manager": vector_store_manager,
            "processing_time": processing_time
        }

def enhanced_search_and_vectorize(
    user_query: str,
    refine_query: bool = True,
    max_results: int = MAX_SEARCH_RESULTS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    save_vector_store: bool = True,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for enhanced_search_and_vectorize_async.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        enhanced_search_and_vectorize_async(
            user_query=user_query,
            refine_query=refine_query,
            max_results=max_results,
            output_dir=output_dir,
            max_concurrent=max_concurrent,
            vector_store_path=vector_store_path,
            llm_model=llm_model,
            save_vector_store=save_vector_store,
            api_key=api_key,
            use_cache=use_cache
        )
    )

# Also fix these functions by using context manager instead of decorator
def perform_vector_search(
    query: str,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    k: int = DEFAULT_RETRIEVAL_RESULTS,
    refine_query: bool = True,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True
) -> List[Any]:
    """
    Search the vector store with GPU acceleration and optional query refinement.
    """
    with track_performance_context("perform_vector_search"):
        search_query = query
        
        # Step 1: Refine query if enabled
        if refine_query:
            try:
                refiner = QueryRefiner(model_name=llm_model, api_key=api_key)
                refined_query = refiner.rewrite_query(query, use_cache=use_cache)
                search_query = refined_query
                logger.info(f"Original query: {query}")
                logger.info(f"Refined query: {refined_query}")
            except Exception as e:
                logger.error(f"Query refinement failed: {str(e)}")
                logger.info("Proceeding with original query.")
        
        # Step 2: Initialize vector store manager with GPU support
        vector_store_manager = VectorStoreManager()
        
        # Step 3: Load vector store
        if not vector_store_manager.load(vector_store_path):
            logger.error(f"Failed to load vector store from {vector_store_path}")
            return []
        
        # Step 4: Search vector store
        results = vector_store_manager.search(search_query, k=k)
        
        return results

def perform_rag_search(
    query: str,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = DEFAULT_RETRIEVAL_RESULTS,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Perform a RAG-based search using the vector store and LLM with GPU optimization.
    """
    with track_performance_context("perform_rag_search"):
        # Initialize vector store manager with GPU support
        vector_store_manager = VectorStoreManager()
        
        # Load vector store
        if not vector_store_manager.load(vector_store_path):
            logger.error(f"Failed to load vector store from {vector_store_path}")
            return {
                "query": query,
                "error": f"Failed to load vector store from {vector_store_path}",
                "timestamp": datetime.now().isoformat()
            }
        
        # Initialize RAG processor
        rag_processor = RAGQueryProcessor(
            vector_store_manager, 
            llm_model=llm_model,
            api_key=api_key
        )
        
        # Process the query
        results = rag_processor.process_query(query, k)
        
        return results

if __name__ == "__main__":
    # Display GPU information
    has_gpu = display_gpu_info()
    
    print(f"\n{'=' * 50}")
    print(f"GPU-Accelerated RAG Search System")
    print(f"{'=' * 50}")
    print(f"Using device: {'GPU' if has_gpu else 'CPU'}")
    
    # Example usage
    print("\nExample 1: Run the full pipeline")
    user_query = "future of renewable energy technologies 2030"
    output_directory = "renewable_energy_content"
    vector_store_directory = "renewable_energy_vector_store"
    
    print(f"Running enhanced search for query: '{user_query}'")
    result = enhanced_search_and_vectorize(
        user_query=user_query,
        refine_query=True,
        max_results=10,
        output_dir=output_directory,
        max_concurrent=5,
        vector_store_path=vector_store_directory,
        llm_model=DEFAULT_LLM_MODEL,
        api_key=DEFAULT_API_KEY
    )
    
    # Example 2: Search the vector store
    print("\nExample 2: Searching the vector store")
    search_query = "innovations in solar power storage"
    search_results = perform_vector_search(
        query=search_query,
        vector_store_path=vector_store_directory,
        k=5,
        refine_query=True
    )
    
    print(f"\nFound {len(search_results)} results for query: '{search_query}'")
    for i, doc in enumerate(search_results):
        print(f"\nResult {i+1} from {doc.metadata.get('source', 'Unknown')}:")
        print(f"Content preview: {doc.page_content[:200]}...")
    
    # Example 3: Perform RAG-based search
    print("\nExample 3: Performing RAG-based search")
    rag_query = "What are the most promising developments in renewable energy?"
    rag_results = perform_rag_search(
        query=rag_query,
        vector_store_path=vector_store_directory,
        k=5
    )
    
    print(f"\nRAG Results for query: '{rag_query}'")
    if "error" in rag_results:
        print(f"Error: {rag_results['error']}")
    else:
        print("\nAnswer:")
        print(rag_results['answer'])