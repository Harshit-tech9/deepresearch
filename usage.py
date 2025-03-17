"""Example usage of the enhanced search and vectorize system."""

import os
import asyncio
from main import enhanced_search_and_vectorize_async, perform_rag_search_sync

# Create base directories
os.makedirs("data", exist_ok=True)
os.makedirs("data/content", exist_ok=True)
os.makedirs("data/vector_stores", exist_ok=True)
os.makedirs("data/cache/search", exist_ok=True)

async def main():
    # Define vector store path correctly
    vector_store_dir = os.path.join("data", "vector_stores", "russia_ukraine")
    
    # First search and vectorize
    result = await enhanced_search_and_vectorize_async(
        user_query="russia ukraine war latest news",
        output_dir="data/content",
        vector_store_path=vector_store_dir,
        max_results=30,
        max_concurrent=5,
        force_flat_index=True,
        refine_query=False,
        use_cache=True
    )

    print("\nSearch and vectorization results:")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
    print(f"Total URLs processed: {result['extraction_results']['total_urls']}")
    print(f"Successfully extracted: {result['extraction_results']['successful_count']}")
    print(f"Failed extractions: {result['extraction_results']['failed_count']}")

    if result['extraction_results']['successful_count'] == 0:
        print("\nNo content was extracted. Please check the logs for details.")
        return

    # Perform a RAG-based search using the SAME vector store directory
    rag_result = perform_rag_search_sync(
        query="Is it promising that the war will end soon?",
        vector_store_path=vector_store_dir,  # Use the same directory
        create_if_missing=False  # Don't create new, use existing
    )

    print("\nRAG Search Results:")
    print("-" * 50)
    if 'answer' in rag_result:
        print("\nAnswer:")
        print(rag_result['answer'])
    elif 'error' in rag_result:
        print("\nError:")
        print(rag_result['error'])
    else:
        print("\nNo answer or error found in response")

if __name__ == "__main__":
    asyncio.run(main())