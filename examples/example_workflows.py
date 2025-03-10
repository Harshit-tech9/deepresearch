"""Example workflows for the GPU-accelerated RAG system."""

import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from main import (
    enhanced_search_and_vectorize,
    perform_vector_search,
    perform_rag_search,
    display_gpu_info
)
from config.settings import DEFAULT_LLM_MODEL, DEFAULT_API_KEY
from src.utils.metrics import track_performance

async def run_batch_queries(
    queries: List[str], 
    output_base_dir: str = "queries_output",
    vector_store_base_dir: str = "queries_vector_store",
    refine_queries: bool = True,
    max_results: int = 20,
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    Run a batch of search queries with GPU acceleration.
    
    Args:
        queries: List of search queries
        output_base_dir: Base directory for outputs
        vector_store_base_dir: Base directory for vector stores
        refine_queries: Whether to refine queries using LLM
        max_results: Maximum number of search results per query
        max_concurrent: Maximum concurrent requests
        
    Returns:
        Dictionary with batch results
    """
    batch_results = {
        "start_time": datetime.now().isoformat(),
        "queries": [],
        "total_queries": len(queries),
        "successful_queries": 0,
        "failed_queries": 0
    }
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: '{query}'")
        
        # Create query-specific directories
        query_hash = str(hash(query) % 10000).zfill(4)
        output_dir = os.path.join(output_base_dir, f"query_{query_hash}")
        vector_store_dir = os.path.join(vector_store_base_dir, f"query_{query_hash}")
        
        try:
            # Run the search and vectorization
            result = await asyncio.to_thread(
                enhanced_search_and_vectorize,
                user_query=query,
                refine_query=refine_queries,
                max_results=max_results,
                output_dir=output_dir,
                max_concurrent=max_concurrent,
                vector_store_path=vector_store_dir,
                llm_model=DEFAULT_LLM_MODEL,
                api_key=DEFAULT_API_KEY
            )
            
            batch_results["queries"].append({
                "query": query,
                "status": "success",
                "output_dir": output_dir,
                "vector_store_dir": vector_store_dir,
                "successful_extractions": result["extraction_results"]["successful_count"],
                "failed_extractions": result["extraction_results"]["failed_count"],
                "processing_time": result["processing_time"]
            })
            
            batch_results["successful_queries"] += 1
            
        except Exception as e:
            batch_results["queries"].append({
                "query": query,
                "status": "failed",
                "error": str(e)
            })
            
            batch_results["failed_queries"] += 1
            print(f"Failed to process query: {query}")
            print(f"Error: {str(e)}")
    
    batch_results["end_time"] = datetime.now().isoformat()
    batch_results["total_time"] = (
        datetime.fromisoformat(batch_results["end_time"]) - 
        datetime.fromisoformat(batch_results["start_time"])
    ).total_seconds()
    
    # Save batch results
    os.makedirs(output_base_dir, exist_ok=True)
    batch_results_path = os.path.join(output_base_dir, "batch_results.json")
    
    with open(batch_results_path, "w") as f:
        import json
        json.dump(batch_results, f, indent=2)
    
    return batch_results

@track_performance
def benchmark_vector_search(
    vector_store_path: str,
    queries: List[str],
    k: int = 5,
    iterations: int = 3
) -> Dict[str, Any]:
    """
    Benchmark vector search performance with GPU acceleration.
    
    Args:
        vector_store_path: Path to the vector store
        queries: List of search queries
        k: Number of results to retrieve
        iterations: Number of iterations per query
        
    Returns:
        Dictionary with benchmark results
    """
    benchmark_results = {
        "start_time": datetime.now().isoformat(),
        "device": "GPU" if os.environ.get("USE_GPU", "True").lower() == "true" else "CPU",
        "queries": [],
        "average_time": 0.0
    }
    
    total_time = 0.0
    total_queries = len(queries) * iterations
    
    for query in queries:
        query_results = {
            "query": query,
            "iterations": [],
            "average_time": 0.0
        }
        
        query_total_time = 0.0
        
        for i in range(iterations):
            start_time = time.time()
            
            results = perform_vector_search(
                query=query,
                vector_store_path=vector_store_path,
                k=k,
                refine_query=False  # Skip refinement for benchmark
            )
            
            end_time = time.time()
            iteration_time = end_time - start_time
            
            query_results["iterations"].append({
                "iteration": i + 1,
                "time": iteration_time,
                "results_count": len(results)
            })
            
            query_total_time += iteration_time
            total_time += iteration_time
            
            print(f"Query: '{query}', Iteration {i+1}/{iterations}, Time: {iteration_time:.4f}s, Results: {len(results)}")
        
        query_results["average_time"] = query_total_time / iterations
        benchmark_results["queries"].append(query_results)
    
    benchmark_results["end_time"] = datetime.now().isoformat()
    benchmark_results["total_time"] = total_time
    benchmark_results["average_time"] = total_time / total_queries
    
    print(f"\nBenchmark Summary:")
    print(f"Total queries: {total_queries} ({len(queries)} unique queries × {iterations} iterations)")
    print(f"Total time: {total_time:.4f}s")
    print(f"Average time per query: {benchmark_results['average_time']:.4f}s")
    
    return benchmark_results

@track_performance
def interactive_demo():
    """Run an interactive demo of the RAG system with GPU acceleration."""
    print("\n" + "=" * 50)
    print("GPU-Accelerated RAG System Interactive Demo")
    print("=" * 50)
    
    # Display GPU information
    has_gpu = display_gpu_info()
    print(f"Using device: {'GPU' if has_gpu else 'CPU'}\n")
    
    while True:
        print("\nOptions:")
        print("1. Search and vectorize")
        print("2. Search vector store")
        print("3. RAG query")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            query = input("\nEnter your search query: ")
            output_dir = input("Enter output directory (default: 'demo_output'): ") or "demo_output"
            vector_store_dir = input("Enter vector store directory (default: 'demo_vector_store'): ") or "demo_vector_store"
            max_results = int(input("Enter maximum results (default: 10): ") or "10")
            
            print(f"\nRunning search and vectorization for query: '{query}'...")
            result = enhanced_search_and_vectorize(
                user_query=query,
                output_dir=output_dir,
                vector_store_path=vector_store_dir,
                max_results=max_results
            )
            
            print(f"\nProcessed {result['extraction_results']['total_urls']} URLs")
            print(f"Successful extractions: {result['extraction_results']['successful_count']}")
            print(f"Failed extractions: {result['extraction_results']['failed_count']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            
        elif choice == "2":
            query = input("\nEnter your search query: ")
            vector_store_dir = input("Enter vector store directory: ")
            k = int(input("Enter number of results (default: 5): ") or "5")
            
            if not os.path.exists(vector_store_dir):
                print(f"Error: Vector store directory '{vector_store_dir}' does not exist")
                continue
            
            print(f"\nSearching vector store for query: '{query}'...")
            results = perform_vector_search(
                query=query,
                vector_store_path=vector_store_dir,
                k=k
            )
            
            print(f"\nFound {len(results)} results:")
            for i, doc in enumerate(results):
                print(f"\nResult {i+1} from {doc.metadata.get('source', 'Unknown')}:")
                print(f"Content preview: {doc.page_content[:200]}...")
            
        elif choice == "3":
            query = input("\nEnter your RAG query: ")
            vector_store_dir = input("Enter vector store directory: ")
            
            if not os.path.exists(vector_store_dir):
                print(f"Error: Vector store directory '{vector_store_dir}' does not exist")
                continue
            
            print(f"\nProcessing RAG query: '{query}'...")
            results = perform_rag_search(
                query=query,
                vector_store_path=vector_store_dir
            )
            
            print("\nRAG Results:")
            if "error" in results:
                print(f"Error: {results['error']}")
            else:
                print("\nAnswer:")
                print(results['answer'])
            
        elif choice == "4":
            print("\nExiting demo. Thank you!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    # Run interactive demo
    interactive_demo()
    
    # Example batch queries
    # asyncio.run(run_batch_queries([
    #     "renewable energy technologies in developing countries",
    #     "machine learning applications in healthcare",
    #     "climate change mitigation strategies",
    #     "blockchain use cases in supply chain"
    # ]))
    
    # Example benchmark
    # benchmark_vector_search(
    #     vector_store_path="renewable_energy_vector_store",
    #     queries=[
    #         "solar power",
    #         "wind energy",
    #         "hydroelectric power",
    #         "geothermal energy"
    #     ],
    #     k=5,
    #     iterations=3
    # )