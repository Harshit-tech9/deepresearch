"""Example usage of the enhanced search and vectorize system."""

from main import enhanced_search_and_vectorize, perform_rag_search

# Run the full pipeline with GPU acceleration
result = enhanced_search_and_vectorize(
    user_query="future of renewable energy technologies 2030",
    output_dir="renewable_energy_content",
    vector_store_path="renewable_energy_vector_store",
    max_results=20,
    max_concurrent=10
)

print(f"\nSearch and vectorization completed in {result['processing_time']:.2f} seconds")
print(f"Processed {result['extraction_results']['total_urls']} URLs")
print(f"Successfully extracted content from {result['extraction_results']['successful_count']} URLs")

# Perform a RAG-based search using the created vector store
rag_result = perform_rag_search(
    query="What are the most promising solar energy storage technologies for 2030?",
    vector_store_path="renewable_energy_vector_store",
    k=5
)

print("\nRAG Search Results:")
print("-" * 50)
print(rag_result['answer'])