"""Search module for DuckDuckGo searches and caching."""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from duckduckgo_search import DDGS

from ..utils.logging import get_logger
from ..utils.metrics import track_performance

logger = get_logger(__name__)

class DuckDuckGoSearcher:
    """Enhanced class to handle DuckDuckGo searches with caching."""
    
    def __init__(self, cache_dir: str = "data/cache/search", cache_ttl: int = 86400):
        """
        Initialize the DuckDuckGo searcher with caching.
        
        Args:
            cache_dir: Directory to store cache files
            cache_ttl: Cache time-to-live in seconds (default: 24 hours)
        """
        self.ddgs = DDGS()
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate a cache key for a search query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _check_cache(self, query: str) -> Optional[List[str]]:
        """Check if a valid cache entry exists for a query."""
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time > timedelta(seconds=self.cache_ttl):
                return None
                
            return data['urls']
        except Exception as e:
            logger.warning(f"Error reading cache: {str(e)}")
            return None
    
    def _save_cache(self, query: str, urls: List[str]) -> None:
        """Save URLs to cache."""
        if not urls:
            return
            
        cache_key = self._get_cache_key(query)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'urls': urls
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Cached {len(urls)} URLs for query: {query}")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")
    
    @track_performance
    def search(self, query: str, max_results: int = 30, use_cache: bool = True, region: str = 'wt-wt', safesearch: str = 'moderate') -> List[str]:
        """
        Search DuckDuckGo and return a list of URLs.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            use_cache: Whether to use cache
            region: Region for search results (default: worldwide)
            safesearch: SafeSearch setting ('on', 'moderate', or 'off')
            
        Returns:
            List of URLs
        """
        # Check cache first
        if use_cache:
            cached = self._check_cache(query)
            if cached:
                logger.info(f"Using {len(cached)} cached results for: {query}")
                return cached[:max_results]
        
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # Get results from DuckDuckGo
            results = []
            try:
                # First try with text search
                for r in self.ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch
                ):
                    if r and isinstance(r, dict):
                        logger.debug(f"Got result: {r}")
                        if 'link' in r:
                            results.append(r['link'])
                        elif 'url' in r:
                            results.append(r['url'])
                    if len(results) >= max_results:
                        break
                
                # If no results, try news search
                if not results:
                    logger.info("No text results, trying news search...")
                    for r in self.ddgs.news(
                        query,
                        region=region,
                        safesearch=safesearch
                    ):
                        if r and isinstance(r, dict):
                            logger.debug(f"Got news result: {r}")
                            if 'link' in r:
                                results.append(r['link'])
                            elif 'url' in r:
                                results.append(r['url'])
                        if len(results) >= max_results:
                            break
            
            except Exception as search_error:
                logger.error(f"Error during search: {str(search_error)}")
                # Try one more time with different parameters
                try:
                    logger.info("Retrying search with different parameters...")
                    for r in self.ddgs.text(
                        query,
                        region='us-en',  # Try US region
                        safesearch='off'  # Try with safesearch off
                    ):
                        if r and isinstance(r, dict):
                            if 'link' in r:
                                results.append(r['link'])
                            elif 'url' in r:
                                results.append(r['url'])
                        if len(results) >= max_results:
                            break
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {str(retry_error)}")
            
            # Remove duplicates while preserving order
            results = list(dict.fromkeys(results))
            
            # Cache results if we got any
            if results and use_cache:
                self._save_cache(query, results)
            
            logger.info(f"Found {len(results)} results for: {query}")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def chat(self, query: str) -> str:
        """Simple wrapper for DuckDuckGo chat."""
        try:
            return self.ddgs.chat(query)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return f"Error: {str(e)}"