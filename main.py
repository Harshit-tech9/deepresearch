"""Main entry point for the application with GPU-optimized pipeline."""

import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

import torch
import nest_asyncio
import json

from config.settings import (
    USE_GPU, GPU_DEVICE, DEFAULT_OUTPUT_DIR, DEFAULT_VECTOR_STORE_PATH,
    MAX_CONCURRENT_REQUESTS, MAX_SEARCH_RESULTS, DEFAULT_RETRIEVAL_RESULTS,
    DEFAULT_LLM_MODEL, DEFAULT_API_KEY, FAISS_INDEX_TYPE
)
from src.utils.logging import get_logger, setup_logger
from src.utils.metrics import performance_monitor, track_performance_context
from src.scraping.improved_scraper import EnhancedWebScraper
from src.search.searcher import DuckDuckGoSearcher
from src.vectorstore.store import VectorStoreManager
from src.vectorstore.manager import initialize_vector_store, create_vector_store_summary
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
    use_cache: bool = True,
    force_flat_index: bool = True  # Always force flat index for reliability
) -> Dict[str, Any]:
    """
    Run a complete search, extraction, and vectorization pipeline with improved robustness.
    """
    with track_performance_context("enhanced_search_and_vectorize"):
        logger.info(f"Starting enhanced search for query: '{user_query}'")
        start_time = time.time()
        
        # Create base directories first
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        
        # Initialize vector store and create directories
        try:
            vector_store_manager, output_dir, vector_dir = initialize_vector_store(
                query=user_query,
                output_base=output_dir,
                vector_base=os.path.dirname(vector_store_path),
                force_flat=force_flat_index
            )
            logger.info(f"Created output directories: {output_dir}, {vector_dir}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            return {
                "query_info": {
                    "original_query": user_query,
                    "refined_query": None,
                    "query_refinement_used": False,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                },
                "extraction_results": {"total_urls": 0, "successful_count": 0, "failed_count": 0, "documents": []},
                "vector_store_manager": None,
                "processing_time": time.time() - start_time
            }
        
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
        try:
            searcher = DuckDuckGoSearcher()
            urls = searcher.search(search_query, max_results=max_results, use_cache=use_cache)
            
            if not urls:
                logger.warning("No URLs returned from search")
                urls = []
            
            logger.info(f"Found {len(urls)} URLs for query: '{search_query}'")
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            urls = []
        
        # Step 3: Extract content
        logger.info(f"Starting extraction of {len(urls)} URLs...")
        
        # Initialize scraper
        scraper = EnhancedWebScraper(output_dir, max_concurrent=max_concurrent)
        
        # Process URLs
        results = await scraper.process_urls(urls)
        
        # Check if we got any successful extractions
        if results['successful_count'] == 0:
            logger.warning("No successful content extractions")
            return {
                "query_info": {
                    "original_query": user_query,
                    "refined_query": search_query if refined else None,
                    "query_refinement_used": refined,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                },
                "extraction_results": results,
                "vector_store_manager": None,
                "processing_time": time.time() - start_time
            }
        
        # Step 4: Chunk documents for better embedding
        documents = results['documents']
        chunked_docs = vector_store_manager.chunk_documents(documents)
        logger.info(f"Chunked {len(documents)} documents into {len(chunked_docs)} chunks")
        
        # Step 5: Add documents to vector store with robust error handling
        try:
            doc_count = vector_store_manager.add_documents(chunked_docs)
            logger.info(f"Added {doc_count} document chunks to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return {
                "query_info": {
                    "original_query": user_query,
                    "refined_query": search_query if refined else None,
                    "query_refinement_used": refined,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": time.time() - start_time
                },
                "extraction_results": results,
                "vector_store_manager": None,
                "processing_time": time.time() - start_time
            }
        
        # Step 6: Save vector store
        if save_vector_store and doc_count > 0:
            try:
                vector_store_manager.save(vector_store_path)
                logger.info(f"Vector store saved to: {os.path.abspath(vector_store_path)}")
            except Exception as e:
                logger.error(f"Error saving vector store: {str(e)}")
        
        # Create summary
        summary = create_vector_store_summary(
            query=user_query,
            output_dir=output_dir,
            vector_dir=vector_dir,
            vector_store_manager=vector_store_manager,
            extraction_results=results
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add processing time to summary
        summary["processing_time"] = processing_time
        
        # Save summary to file
        try:
            with open(os.path.join(output_dir, 'query_info.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving summary: {str(e)}")
        
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
        
        # Return the results
        return {
            "query_info": summary,
            "extraction_results": results,
            "vector_store_manager": vector_store_manager if doc_count > 0 else None,
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
    use_cache: bool = True,
    force_flat_index: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for enhanced_search_and_vectorize_async with improved error handling.
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
            use_cache=use_cache,
            force_flat_index=force_flat_index
        )
    )

def perform_vector_search(
    query: str,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    k: int = DEFAULT_RETRIEVAL_RESULTS,
    refine_query: bool = True,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True,
    create_if_missing: bool = False  # Add option to create vector store if missing
) -> List[Any]:
    """
    Search the vector store with robust error handling.
    
    Args:
        query: The search query
        vector_store_path: Path to the vector store
        k: Number of results to retrieve
        refine_query: Whether to refine the query
        llm_model: LLM model for query refinement
        api_key: API key
        use_cache: Whether to use caching
        create_if_missing: Whether to create a new vector store if missing
        
    Returns:
        Search results
    """
    with track_performance_context("perform_vector_search"):
        search_query = query
        
        # Check if vector store exists
        if not os.path.exists(vector_store_path):
            logger.warning(f"Vector store path {vector_store_path} does not exist")
            
            if create_if_missing:
                logger.info(f"Creating new vector store at {vector_store_path}")
                try:
                    result = enhanced_search_and_vectorize(
                        user_query=query,
                        refine_query=refine_query,
                        vector_store_path=vector_store_path,
                        llm_model=llm_model,
                        api_key=api_key,
                        use_cache=use_cache,
                        force_flat_index=True  # Use Flat index for new stores
                    )
                    
                    if result["extraction_results"]["successful_count"] == 0:
                        logger.warning("No documents extracted for the query")
                        return []
                    
                    # Continue with the search using the new vector store
                    logger.info("New vector store created, continuing with search")
                except Exception as e:
                    logger.error(f"Failed to create new vector store: {str(e)}")
                    return []
            else:
                return []
        
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
        
        # Step 2: Initialize vector store manager
        vector_store_manager = VectorStoreManager()
        
        # Step 3: Load vector store with proper error handling
        if not vector_store_manager.load(vector_store_path):
            logger.error(f"Failed to load vector store from {vector_store_path}")
            return []
        
        # Step 4: Search vector store
        try:
            results = vector_store_manager.search(search_query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []

async def perform_rag_search(
    query: str,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = DEFAULT_RETRIEVAL_RESULTS,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True,
    create_if_missing: bool = True  # Default to True to auto-create if missing
) -> Dict[str, Any]:
    """
    Perform a RAG-based search with robust error handling.
    """
    with track_performance_context("perform_rag_search"):
        try:
            # Normalize path to use correct separators
            vector_store_path = os.path.normpath(vector_store_path)
            logger.info(f"Using vector store path: {vector_store_path}")
            
            # Check if vector store exists by looking for index.faiss
            index_path = os.path.join(vector_store_path, "index.faiss")
            if not os.path.exists(index_path):
                logger.warning(f"Vector store index not found at {index_path}")
                
                if create_if_missing:
                    logger.info(f"Creating new vector store at {vector_store_path}")
                    try:
                        # Initialize vector store and directories
                        vector_store_manager, output_dir, _ = initialize_vector_store(
                            query=query,
                            output_base=os.path.join(os.path.dirname(vector_store_path), "content"),
                            vector_base=os.path.dirname(vector_store_path),
                            force_flat=True
                        )
                        
                        # Create the vector store using enhanced search
                        result = await enhanced_search_and_vectorize_async(
                            user_query=query,
                            refine_query=True,
                            output_dir=output_dir,
                            vector_store_path=vector_store_path,
                            llm_model=llm_model,
                            api_key=api_key,
                            use_cache=use_cache,
                            force_flat_index=True
                        )
                        
                        if result["vector_store_manager"] is None:
                            logger.warning("Failed to create vector store")
                            return {
                                "query": query,
                                "error": "Failed to create vector store - no documents found or extraction failed",
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        # Use the newly created vector store manager
                        vector_store_manager = result["vector_store_manager"]
                        logger.info("New vector store created, continuing with search")
                    except Exception as e:
                        logger.error(f"Failed to create new vector store: {str(e)}")
                        return {
                            "query": query,
                            "error": f"Failed to create vector store: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        }
                else:
                    return {
                        "query": query,
                        "error": f"Vector store index not found at {index_path}",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Initialize vector store manager with Flat index
            vector_store_manager = VectorStoreManager(index_type="Flat")
            
            # Load vector store with proper error handling
            try:
                if not vector_store_manager.load(vector_store_path):
                    logger.error(f"Failed to load vector store from {vector_store_path}")
                    return {
                        "query": query,
                        "error": f"Failed to load vector store from {vector_store_path}",
                        "timestamp": datetime.now().isoformat()
                    }
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                return {
                    "query": query,
                    "error": f"Error loading vector store: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Initialize RAG processor with robust error handling
            try:
                rag_processor = RAGQueryProcessor(
                    vector_store_manager, 
                    llm_model=llm_model,
                    api_key=api_key
                )
                
                # Process the query
                results = rag_processor.process_query(query, k)
                
                return results
            except Exception as e:
                logger.error(f"Error in RAG processing: {str(e)}")
                return {
                    "query": query,
                    "error": f"RAG processing error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Unexpected error in RAG search: {str(e)}")
            return {
                "query": query,
                "error": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

def perform_rag_search_sync(
    query: str,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    k: int = DEFAULT_RETRIEVAL_RESULTS,
    api_key: str = DEFAULT_API_KEY,
    use_cache: bool = True,
    create_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for perform_rag_search.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        perform_rag_search(
            query=query,
            vector_store_path=vector_store_path,
            llm_model=llm_model,
            k=k,
            api_key=api_key,
            use_cache=use_cache,
            create_if_missing=create_if_missing
        )
    )
