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
    
    def __init__(self, cache_dir: str = "search_cache", cache_ttl: int = 86400):
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
    
    def _get_cache_key(self, query: str, max_results: int) -> str:
        """
        Generate a cache key for a search query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Cache key
        """
        query_hash = hashlib.md5(f"{query}_{max_results}".encode()).hexdigest()
        return query_hash
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cache file path
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _check_cache(self, cache_key: str) -> Optional[List[str]]:
        """
        Check if a valid cache entry exists for a key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached URLs if valid, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(seconds=self.cache_ttl):
                logger.info(f"Cache expired for key: {cache_key}")
                return None
            
            return cache_data['urls']
        except Exception as e:
            logger.warning(f"Error reading cache: {str(e)}")
            return None
    
    def _save_cache(self, cache_key: str, urls: List[str]) -> None:
        """
        Save URLs to cache.
        
        Args:
            cache_key: Cache key
            urls: List of URLs to cache
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'urls': urls
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Saved {len(urls)} URLs to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")
    
    @track_performance
    def search(self, query: str, max_results: int = 30, use_cache: bool = True) -> List[str]:
        """
        Search DuckDuckGo and return a list of URLs with optional caching.
        
        Args:
            query: The search query
            max_results: Maximum number of results to retrieve
            use_cache: Whether to use cache
            
        Returns:
            List of URLs from search results
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = self._get_cache_key(query, max_results)
            cached_urls = self._check_cache(cache_key)
            
            if cached_urls:
                logger.info(f"Using cached results for query: {query}")
                return cached_urls
        
        logger.info(f"Performing search for query: {query}")
        try:
            results = self.ddgs.text(query, max_results=max_results)
            
            # Extract URLs from results
            urls = []
            for result in results:
                if 'href' in result:
                    urls.append(result['href'])
            
            # Save to cache if enabled
            if use_cache and urls:
                cache_key = self._get_cache_key(query, max_results)
                self._save_cache(cache_key, urls)
            
            logger.info(f"Found {len(urls)} results for query: {query}")
            return urls
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    def chat(self, query: str, model: str = 'claude-3-haiku') -> str:
        """
        Perform a chat query using DuckDuckGo.
        
        Args:
            query: The chat query
            model: The model to use for chat
            
        Returns:
            Chat response
        """
        try:
            response = self.ddgs.chat(query, model=model)
            return response
        except Exception as e:
            logger.error(f"Error with DuckDuckGo chat: {str(e)}")
            return f"Error: {str(e)}"