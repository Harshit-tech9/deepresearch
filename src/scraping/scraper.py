"""Enhanced web scraping module with improved concurrency and error handling."""

import asyncio
import aiohttp
import os
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from langchain_core.documents import Document

from ..utils.logging import get_logger
from ..utils.metrics import track_performance

logger = get_logger(__name__)

class WebScraper:
    """Enhanced class to handle web scraping operations with improved concurrency."""
    
    def __init__(self, output_dir: str = "extracted_content", max_concurrent: int = 15):
        """Initialize the scraper with output directory and concurrency limit."""
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for better organization
        self.text_dir = os.path.join(output_dir, "text")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Track processed URLs to avoid duplicates
        self.processed_urls: Set[str] = set()
    
    @staticmethod
    async def fetch_content(
        session: aiohttp.ClientSession, 
        url: str, 
        semaphore: asyncio.Semaphore, 
        timeout: int = 30,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Fetch content from a URL with retry logic and improved error handling.
        
        Args:
            session: aiohttp client session
            url: URL to fetch
            semaphore: Semaphore to limit concurrent requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            Dictionary with fetched content and metadata
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    async with session.get(url, timeout=timeout, headers=headers) as response:
                        if response.status == 200:
                            content_type = response.headers.get('Content-Type', '')
                            if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
                                html = await response.text()
                                return {
                                    'url': url,
                                    'status': response.status,
                                    'headers': dict(response.headers),
                                    'html': html,
                                    'timestamp': datetime.now().isoformat()
                                }
                            else:
                                logger.warning(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
                                return {
                                    'url': url,
                                    'status': response.status,
                                    'error': f'Non-HTML content: {content_type}',
                                    'timestamp': datetime.now().isoformat()
                                }
                        elif response.status in (429, 503):  # Rate limiting or service unavailable
                            if attempt < max_retries:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.warning(f"Rate limited or service unavailable for {url}. Retrying in {wait_time:.2f}s...")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        # For other status codes, return error
                        return {
                            'url': url,
                            'status': response.status,
                            'error': f'HTTP Error: {response.status}',
                            'timestamp': datetime.now().isoformat()
                        }
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"Timeout for {url}. Retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'url': url,
                    'status': None,
                    'error': 'Timeout Error',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    'url': url,
                    'status': None,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # If we've exhausted retries
        return {
            'url': url,
            'status': None,
            'error': 'Maximum retries exceeded',
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def extract_text_from_html(html: str) -> str:
        """
        Extract readable text content from HTML with improved parsing.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe']):
                element.extract()
            
            # Focus on content areas
            content_areas = []
            for tag in ['article', 'main', 'div.content', 'div.main', 'section']:
                elements = soup.select(tag)
                if elements:
                    content_areas.extend(elements)
            
            # If content areas were found, use them, otherwise use the whole body
            if content_areas:
                text = ' '.join(area.get_text(separator=' ') for area in content_areas)
            else:
                text = soup.get_text(separator=' ')
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove very short lines (often menu items or buttons)
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
            text = ' '.join(lines)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return ""
    
    @staticmethod
    def get_safe_filename(url: str) -> str:
        """
        Create a safe filename from URL with improved handling.
        
        Args:
            url: URL
            
        Returns:
            Safe filename
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('.', '_')
            path = parsed.path.replace('/', '_')
            
            if path:
                filename = f"{domain}{path[:50]}"
            else:
                filename = domain
            
            # Remove any invalid characters
            filename = re.sub(r'[^\w\-_]', '', filename)
            
            # Ensure filename is not too long
            if len(filename) > 100:
                filename = filename[:100]
            
            # Add hash to ensure uniqueness
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"{filename}_{url_hash}"
            
            return filename
        except Exception as e:
            logger.error(f"Error creating safe filename from URL: {str(e)}")
            # Fallback to a hash of the URL
            import hashlib
            return hashlib.md5(url.encode()).hexdigest()
    
    async def process_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Process a list of URLs concurrently and save their content.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary with results
        """
        # Deduplicate URLs
        unique_urls = [url for url in urls if url not in self.processed_urls]
        logger.info(f"Processing {len(unique_urls)} unique URLs out of {len(urls)} total URLs")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Keep track of successful and failed URLs
        results = {
            'successful': [],
            'failed': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'documents': []  # Store Document objects for vector store
        }
        
        # Configure aiohttp client session with connection pooling
        conn = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            # Create tasks for all URLs
            tasks = [self.fetch_content(session, url, semaphore) for url in unique_urls]
            
            # Process as they complete
            for task in asyncio.as_completed(tasks):
                result = await task
                url = result['url']
                self.processed_urls.add(url)
                
                if 'error' in result and 'html' not in result:
                    results['failed'].append({
                        'url': url,
                        'error': result['error'],
                        'timestamp': result['timestamp']
                    })
                    logger.warning(f"Failed to fetch: {url} - {result['error']}")
                    continue
                
                if 'html' in result:
                    # Extract text content
                    text = self.extract_text_from_html(result['html'])
                    
                    if not text or len(text) < 100:  # Skip pages with little content
                        results['failed'].append({
                            'url': url,
                            'error': 'Insufficient content',
                            'timestamp': result['timestamp']
                        })
                        logger.warning(f"Insufficient content: {url}")
                        continue
                    
                    # Create safe filename
                    filename = self.get_safe_filename(url)
                    
                    # Save text content
                    text_file = os.path.join(self.text_dir, f"{filename}.txt")
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Save metadata
                    metadata = {
                        'url': url,
                        'status': result['status'],
                        'timestamp': result['timestamp'],
                        'headers': result['headers'],
                        'text_length': len(text),
                        'filename': f"{filename}.txt"
                    }
                    
                    metadata_file = os.path.join(self.metadata_dir, f"{filename}_metadata.json")
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Create Document object for vector store
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': url,
                            'timestamp': result['timestamp'],
                            'filename': f"{filename}.txt"
                        }
                    )
                    
                    # Add to documents list
                    results['documents'].append(doc)
                    
                    results['successful'].append({
                        'url': url,
                        'text_file': text_file,
                        'metadata_file': metadata_file,
                        'text_length': len(text)
                    })
                    
                    logger.info(f"Successfully processed: {url}")
        
        # Save overall results
        results['end_time'] = datetime.now().isoformat()
        results['total_urls'] = len(unique_urls)
        results['successful_count'] = len(results['successful'])
        results['failed_count'] = len(results['failed'])
        
        summary_path = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in results.items() if k != 'documents'}, f, indent=2)
        
        logger.info(f"Extraction complete: {results['successful_count']} successful, {results['failed_count']} failed")
        return results