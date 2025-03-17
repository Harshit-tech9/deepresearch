"""Utility for dynamically creating vector stores."""

import os
import sys
import argparse
import hashlib
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import enhanced_search_and_vectorize
from config.settings import DEFAULT_LLM_MODEL, DEFAULT_API_KEY

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

def create_vector_store(
    query,
    output_base="content",
    vector_base="vector_stores",
    max_results=30,
    refine_query=True,
    force_flat=True
):
    """
    Create a vector store for a query dynamically.
    
    Args:
        query: The search query
        output_base: Base directory for content
        vector_base: Base directory for vector stores
        max_results: Maximum search results
        refine_query: Whether to refine the query
        force_flat: Force using Flat index type
        
    Returns:
        Dictionary with results
    """
    # Create safe directory names
    output_dir = get_safe_path(query, output_base)
    vector_dir = get_safe_path(query, vector_base)
    
    # Ensure base directories exist
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(vector_base, exist_ok=True)
    
    print(f"Creating vector store for query: '{query}'")
    print(f"Output directory: {output_dir}")
    print(f"Vector store directory: {vector_dir}")
    
    start_time = datetime.now()
    
    # Run the search and vectorization
    result = enhanced_search_and_vectorize(
        user_query=query,
        refine_query=refine_query,
        max_results=max_results,
        output_dir=output_dir,
        vector_store_path=vector_dir,
        force_flat_index=force_flat
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nVector store creation complete in {duration:.2f} seconds")
    print(f"Processed {result['extraction_results']['total_urls']} URLs")
    print(f"Successful extractions: {result['extraction_results']['successful_count']}")
    print(f"Failed extractions: {result['extraction_results']['failed_count']}")
    
    doc_count = 0
    if result['vector_store_manager']:
        doc_count = len(result['vector_store_manager'].vector_store.docstore._dict)
        print(f"Documents in vector store: {doc_count}")
    else:
        print("No vector store manager was created")
    
    # Save a summary file
    summary = {
        "query": query,
        "output_dir": output_dir,
        "vector_dir": vector_dir,
        "created_at": end_time.isoformat(),
        "duration_seconds": duration,
        "urls_processed": result['extraction_results']['total_urls'],
        "successful_extractions": result['extraction_results']['successful_count'],
        "failed_extractions": result['extraction_results']['failed_count'],
        "documents_count": doc_count,
        "has_vector_store": bool(result['vector_store_manager'])
    }
    
    summary_file = os.path.join(vector_base, "vector_stores_summary.txt")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- {datetime.now().isoformat()} ---\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
    
    return {
        "summary": summary,
        "result": result,
        "vector_path": vector_dir
    }

def main():
    """Command-line interface for creating vector stores."""
    parser = argparse.ArgumentParser(description="Create a vector store for a query")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum search results")
    parser.add_argument("--output-base", default="content", help="Base directory for content")
    parser.add_argument("--vector-base", default="vector_stores", help="Base directory for vector stores")
    parser.add_argument("--no-refine", action="store_false", dest="refine", help="Don't refine the query")
    parser.add_argument("--use-ivf", action="store_false", dest="force_flat", help="Allow IVFFlat index")
    
    args = parser.parse_args()
    
    create_vector_store(
        query=args.query,
        output_base=args.output_base,
        vector_base=args.vector_base,
        max_results=args.max_results,
        refine_query=args.refine,
        force_flat=args.force_flat
    )

if __name__ == "__main__":
    main()