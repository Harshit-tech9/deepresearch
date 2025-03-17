"""Enhanced web scraping module with improved success rate."""

import asyncio
import aiohttp
import os
import re
import json
import random
import time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from langchain_core.documents import Document

# For logging
import logging
logger = logging.getLogger(__name__)

class EnhancedWebScraper:
    """Enhanced class to handle web scraping operations with higher success rate."""
    
    # List of user agents to rotate through (mimics different browsers and devices)
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (iPad; CPU OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    ]
    
    # Accepted content types
    ACCEPTED_CONTENT_TYPES = [
        'text/html', 
        'application/xhtml+xml', 
        'text/plain',
        'application/xml',
        'text/xml',
        'application/json',  # For API responses
    ]
    
    # Common bot detection strings to avoid in URLs
    BOT_DETECTION_STRINGS = [
        'captcha', 
        'robot', 
        'challenge',
        'security-check',
        'blocked',
        'verification'
    ]
    
    def __init__(self, 
                 output_dir: str = "extracted_content", 
                 max_concurrent: int = 5,  # Reduced concurrency to avoid rate limits
                 min_text_length: int = 50,  # Reduced minimum text length to consider content valid
                 retry_statuses: List[int] = None,
                 follow_redirects: bool = True,
                 respect_robots_txt: bool = True):
        """Initialize the enhanced scraper with improved settings."""
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.min_text_length = min_text_length
        self.retry_statuses = retry_statuses or [429, 503, 502, 500, 408]
        self.follow_redirects = follow_redirects
        self.respect_robots_txt = respect_robots_txt
        
        # Create directories
        self.text_dir = os.path.join(output_dir, "text")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        self.error_dir = os.path.join(output_dir, "errors")  # New directory for error details
        
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        
        # Track processed URLs and errors
        self.processed_urls: Set[str] = set()
        self.error_details: Dict[str, List[Dict]] = {}
        
        # For tracking the types of errors
        self.error_counts: Dict[str, int] = {}
    
    def get_random_user_agent(self):
        """Get a random user agent from the list."""
        return random.choice(self.USER_AGENTS)
    
    def should_retry(self, status_code: Optional[int]) -> bool:
        """Determine if a request should be retried based on status code."""
        if status_code is None:
            return True  # Retry on connection errors (no status code)
        return status_code in self.retry_statuses
    
    def detect_bot_trap(self, url: str) -> bool:
        """Check if URL might be a bot trap."""
        url_lower = url.lower()
        return any(trap in url_lower for trap in self.BOT_DETECTION_STRINGS)
    
    def get_delay(self, domain: str) -> float:
        """Get appropriate delay for a domain to avoid rate limiting."""
        # Higher delay for known strict sites
        strict_sites = ['google.com', 'linkedin.com', 'facebook.com', 'twitter.com']
        base_delay = 2.0  # Base delay in seconds
        
        for site in strict_sites:
            if site in domain:
                return base_delay * 2
        
        return base_delay
    
    async def fetch_content(self,
                           session: aiohttp.ClientSession, 
                           url: str, 
                           semaphore: asyncio.Semaphore, 
                           timeout: int = 30,
                           max_retries: int = 3,
                           retry_delay_base: float = 1.0,
                           follow_redirects: bool = True) -> Dict[str, Any]:
        """
        Fetch content from a URL with enhanced retry logic and error handling.
        """
        # Parse domain for domain-specific handling
        domain = urlparse(url).netloc
        
        # Skip known bot traps
        if self.detect_bot_trap(url):
            return {
                'url': url,
                'status': None,
                'error': 'Skipped potential bot trap URL',
                'timestamp': datetime.now().isoformat()
            }
        
        # Build headers with random user agent
        headers = {
            'User-Agent': self.get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'https://www.google.com/'  # Common referer to appear more natural
        }
        
        # Add random delay before request to avoid rate limiting patterns
        # Use a domain-specific delay to respect sites with stricter rate limits
        await asyncio.sleep(self.get_delay(domain) * random.uniform(0.8, 1.2))
        
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    # Configure client options
                    request_kwargs = {
                        'url': url, 
                        'timeout': aiohttp.ClientTimeout(total=timeout),
                        'headers': headers,
                        'allow_redirects': follow_redirects,
                        'ssl': False  # More permissive SSL to handle various configs
                    }
                    
                    async with session.get(**request_kwargs) as response:
                        # Track the status for debugging
                        if str(response.status) not in self.error_counts:
                            self.error_counts[str(response.status)] = 0
                        self.error_counts[str(response.status)] += 1
                        
                        # Log response info for debugging
                        logger.debug(f"Attempt {attempt+1} for {url}: Status {response.status}")
                        
                        # Handle successful responses
                        if response.status == 200:
                            # Check content type
                            content_type = response.headers.get('Content-Type', '').lower()
                            is_accepted_type = any(accepted in content_type for accepted in self.ACCEPTED_CONTENT_TYPES)
                            
                            if is_accepted_type or 'text/' in content_type:
                                try:
                                    html = await response.text()
                                    
                                    # Basic validation - check if it's actually HTML content
                                    if len(html) < 50 or not ('<html' in html.lower() or '<body' in html.lower()):
                                        logger.warning(f"Retrieved content doesn't appear to be valid HTML: {url}")
                                        
                                        if attempt < max_retries:
                                            logger.info(f"Retrying {url} due to invalid HTML...")
                                            await asyncio.sleep(retry_delay_base * (2 ** attempt))
                                            continue
                                        else:
                                            return {
                                                'url': url,
                                                'status': response.status,
                                                'error': 'Retrieved content is not valid HTML',
                                                'timestamp': datetime.now().isoformat()
                                            }
                                    
                                    return {
                                        'url': url,
                                        'status': response.status,
                                        'headers': dict(response.headers),
                                        'html': html,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                except UnicodeDecodeError as e:
                                    # Try with different encoding if default fails
                                    try:
                                        html = await response.read()
                                        html = html.decode('utf-8', errors='replace')
                                        return {
                                            'url': url,
                                            'status': response.status,
                                            'headers': dict(response.headers),
                                            'html': html,
                                            'timestamp': datetime.now().isoformat()
                                        }
                                    except Exception as e2:
                                        logger.error(f"Encoding error on {url}: {str(e)} -> {str(e2)}")
                                        if attempt < max_retries:
                                            continue
                                        return {
                                            'url': url,
                                            'status': response.status,
                                            'error': f'Encoding error: {str(e2)}',
                                            'timestamp': datetime.now().isoformat()
                                        }
                            else:
                                logger.warning(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
                                # For certain content types, try to extract text
                                if 'application/pdf' in content_type:
                                    return {
                                        'url': url,
                                        'status': response.status,
                                        'error': f'PDF content not supported: {content_type}',
                                        'timestamp': datetime.now().isoformat()
                                    }
                                elif 'image/' in content_type:
                                    return {
                                        'url': url,
                                        'status': response.status,
                                        'error': f'Image content not supported: {content_type}',
                                        'timestamp': datetime.now().isoformat()
                                    }
                                else:
                                    return {
                                        'url': url,
                                        'status': response.status,
                                        'error': f'Unsupported content type: {content_type}',
                                        'timestamp': datetime.now().isoformat()
                                    }
                        
                        # Handle redirects explicitly, especially if aiohttp's auto-redirect isn't working
                        elif response.status in (301, 302, 303, 307, 308) and not follow_redirects:
                            redirect_url = response.headers.get('Location')
                            if redirect_url:
                                # Handle relative URLs
                                if redirect_url.startswith('/'):
                                    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                                    redirect_url = urljoin(base_url, redirect_url)
                                
                                logger.info(f"Following redirect: {url} -> {redirect_url}")
                                return await self.fetch_content(
                                    session, redirect_url, semaphore, timeout, max_retries, retry_delay_base, True
                                )
                        
                        # Handle retryable status codes
                        elif self.should_retry(response.status) and attempt < max_retries:
                            wait_time = retry_delay_base * (2 ** attempt) * random.uniform(0.8, 1.2)
                            logger.warning(f"Retryable status {response.status} for {url}. Waiting {wait_time:.2f}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # For other status codes, return error
                        else:
                            return {
                                'url': url,
                                'status': response.status,
                                'error': f'HTTP Error: {response.status}',
                                'timestamp': datetime.now().isoformat()
                            }
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} on attempt {attempt+1}/{max_retries+1}")
                if attempt < max_retries:
                    wait_time = retry_delay_base * (2 ** attempt) * random.uniform(0.8, 1.2)
                    await asyncio.sleep(wait_time)
                    continue
                return {
                    'url': url,
                    'status': None,
                    'error': 'Timeout Error',
                    'timestamp': datetime.now().isoformat()
                }
            except (aiohttp.ClientError, aiohttp.ClientConnectorError) as e:
                error_type = type(e).__name__
                if error_type not in self.error_counts:
                    self.error_counts[error_type] = 0
                self.error_counts[error_type] += 1
                
                logger.warning(f"{error_type} for {url} on attempt {attempt+1}/{max_retries+1}: {str(e)}")
                if attempt < max_retries:
                    wait_time = retry_delay_base * (2 ** attempt) * random.uniform(0.8, 1.2)
                    await asyncio.sleep(wait_time)
                    continue
                return {
                    'url': url,
                    'status': None,
                    'error': f'{error_type}: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_type = type(e).__name__
                if error_type not in self.error_counts:
                    self.error_counts[error_type] = 0
                self.error_counts[error_type] += 1
                
                logger.error(f"Unexpected error for {url}: {error_type}: {str(e)}")
                return {
                    'url': url,
                    'status': None,
                    'error': f'{error_type}: {str(e)}',
                    'timestamp': datetime.now().isoformat()
                }
        
        # If we've exhausted retries
        return {
            'url': url,
            'status': None,
            'error': 'Maximum retries exceeded',
            'timestamp': datetime.now().isoformat()
        }
    
    def extract_text_from_html(self, html: str, url: str = "") -> str:
        """
        Extract readable text content from HTML with improved algorithms.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form', 'iframe', 'noscript']):
                element.extract()
            
            # Check for common content containers
            content_tags = [
                'article', 'main', 'div[class*="content"]', 'div[class*="article"]',
                'div[class*="main"]', 'div[class*="post"]', 'div[id*="content"]',
                'div[id*="article"]', 'div[id*="main"]', 'div[id*="post"]',
                'section', 'div[class*="body"]', '.post-content'
            ]
            
            # Try to find main content container
            content = None
            for tag in content_tags:
                elements = soup.select(tag)
                if elements:
                    # Find the content container with the most text
                    largest_element = max(elements, key=lambda e: len(e.get_text(strip=True)), default=None)
                    if largest_element and len(largest_element.get_text(strip=True)) > self.min_text_length:
                        content = largest_element
                        break
            
            # If no main content found, use the whole body
            if not content or len(content.get_text(strip=True)) < self.min_text_length:
                content = soup.body or soup
            
            # Extract paragraphs with decent text content
            paragraphs = []
            for p in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li', 'td', 'div']):
                text = p.get_text(strip=True)
                # Only include paragraphs with meaningful content
                if len(text) > 15 and not text.isupper():  # Skip all-uppercase (often menus)
                    paragraphs.append(text)
            
            # If we found paragraphs, join them with newlines
            if paragraphs:
                return '\n\n'.join(paragraphs)
            
            # Fallback: get all text with basic cleaning
            text = content.get_text(separator=' ')
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove very short lines
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
            text = '\n'.join(lines)
            
            # If text is still too short, try to extract from the whole page
            if len(text) < self.min_text_length:
                text = soup.get_text(separator=' ')
                text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            # Save the HTML for debugging
            if url:
                debug_path = os.path.join(self.error_dir, f"extraction_error_{urlparse(url).netloc}.html")
                try:
                    with open(debug_path, 'w', encoding='utf-8') as f:
                        f.write(html[:10000])  # Save first 10KB
                except Exception as write_err:
                    logger.error(f"Error saving debug HTML: {str(write_err)}")
            return ""
    
    def get_safe_filename(self, url: str) -> str:
        """
        Create a safe filename from URL with improved handling.
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
        Process a list of URLs concurrently with improved error handling.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            Dictionary containing results and statistics
        """
        if not urls:
            logger.warning("No URLs provided to process")
            return {
                "total_urls": 0,
                "successful_count": 0,
                "failed_count": 0,
                "documents": [],
                "errors": {}
            }
            
        logger.info(f"Starting to process {len(urls)} URLs with max concurrency of {self.max_concurrent}")
        
        # Initialize statistics
        total_urls = len(urls)
        successful_count = 0
        failed_count = 0
        documents = []
        errors = {}
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Configure aiohttp session with longer timeout and keep-alive
        timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, force_close=False, enable_cleanup_closed=True)
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # Create tasks for all URLs
            tasks = []
            for url in urls:
                if url not in self.processed_urls:  # Skip already processed URLs
                    tasks.append(self.fetch_content(session, url, semaphore))
                    self.processed_urls.add(url)
            
            if not tasks:
                logger.warning("No new URLs to process (all URLs have been processed before)")
                return {
                    "total_urls": total_urls,
                    "successful_count": 0,
                    "failed_count": 0,
                    "documents": [],
                    "errors": {}
                }
            
            # Process URLs concurrently and gather results
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Task failed with exception: {str(result)}")
                        failed_count += 1
                        continue
                        
                    if not result:
                        logger.warning("Empty result received")
                        failed_count += 1
                        continue
                        
                    url = result.get('url', 'unknown_url')
                    
                    if 'error' in result:
                        logger.warning(f"Error processing {url}: {result['error']}")
                        failed_count += 1
                        errors[url] = result['error']
                        continue
                        
                    if 'html' not in result:
                        logger.warning(f"No HTML content in result for {url}")
                        failed_count += 1
                        errors[url] = "No HTML content in response"
                        continue
                    
                    try:
                        # Extract text from HTML
                        text = self.extract_text_from_html(result['html'], url)
                        
                        if len(text.strip()) < self.min_text_length:
                            logger.warning(f"Extracted text too short for {url}")
                            failed_count += 1
                            errors[url] = "Extracted text too short"
                            continue
                        
                        # Create document
                        metadata = {
                            'source': url,
                            'timestamp': result['timestamp'],
                            'title': self.extract_title(result['html']) or url,
                        }
                        
                        document = Document(
                            page_content=text,
                            metadata=metadata
                        )
                        
                        documents.append(document)
                        successful_count += 1
                        
                        # Save content and metadata
                        self.save_content(url, text, metadata)
                        
                    except Exception as e:
                        logger.error(f"Error processing content from {url}: {str(e)}")
                        failed_count += 1
                        errors[url] = f"Content processing error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Error during concurrent processing: {str(e)}")
                return {
                    "total_urls": total_urls,
                    "successful_count": successful_count,
                    "failed_count": failed_count + len(tasks) - successful_count,
                    "documents": documents,
                    "errors": {**errors, "concurrent_processing_error": str(e)}
                }
        
        # Log final statistics
        logger.info(f"Processing completed:")
        logger.info(f"Total URLs: {total_urls}")
        logger.info(f"Successfully processed: {successful_count}")
        logger.info(f"Failed: {failed_count}")
        if errors:
            logger.info(f"Error types encountered: {list(set(errors.values()))}")
        
        return {
            "total_urls": total_urls,
            "successful_count": successful_count,
            "failed_count": failed_count,
            "documents": documents,
            "errors": errors
        }
    
    def extract_title(self, html: str) -> Optional[str]:
        """
        Extract the title from HTML content.
        
        Args:
            html: HTML content to extract title from
            
        Returns:
            Title string if found, None otherwise
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to get title from meta tags first
            meta_title = soup.find('meta', property='og:title')
            if meta_title and meta_title.get('content'):
                return meta_title['content'].strip()
            
            # Try article title
            article_title = soup.find('h1', class_=lambda x: x and 'title' in x.lower())
            if article_title:
                return article_title.get_text().strip()
            
            # Fall back to regular title tag
            title_tag = soup.title
            if title_tag:
                return title_tag.string.strip()
            
            return None
        except Exception as e:
            logger.warning(f"Error extracting title: {str(e)}")
            return None
    
    def save_content(self, url: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Save extracted content and metadata to files.
        
        Args:
            url: Source URL
            text: Extracted text content
            metadata: Metadata dictionary
        """
        try:
            # Create safe filename from URL
            filename = self.get_safe_filename(url)
            
            # Save text content
            text_file = os.path.join(self.text_dir, f"{filename}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Add file information to metadata
            metadata.update({
                'text_file': text_file,
                'text_length': len(text),
                'extraction_time': datetime.now().isoformat()
            })
            
            # Save metadata
            metadata_file = os.path.join(self.metadata_dir, f"{filename}_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved content and metadata for {url}")
            
        except Exception as e:
            logger.error(f"Error saving content for {url}: {str(e)}")
            # Don't raise the exception - we don't want to fail the whole process
            # The document is already created and will be returned in the results