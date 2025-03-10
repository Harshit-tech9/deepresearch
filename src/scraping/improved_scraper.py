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
        Process a list of URLs concurrently with improved error handling and detailed reporting.
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
            'documents': [],  # Store Document objects for vector store
            'error_stats': {}  # Store error statistics
        }
        
        # Configure aiohttp client session with improved settings
        conn = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            ssl=False,  # More permissive SSL handling
            ttl_dns_cache=300  # Cache DNS lookups
        )
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        
        # Create session with improved settings
        async with aiohttp.ClientSession(
            connector=conn, 
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar()  # Handle cookies for better site compatibility
        ) as session:
            # Process URLs in chunks to avoid overwhelming servers
            chunk_size = 5  # Process 5 URLs at a time
            for i in range(0, len(unique_urls), chunk_size):
                chunk = unique_urls[i:i+chunk_size]
                logger.info(f"Processing URL chunk {i//chunk_size + 1}/{(len(unique_urls) - 1) // chunk_size + 1}")
                
                # Create tasks for this chunk
                tasks = [self.fetch_content(session, url, semaphore, follow_redirects=self.follow_redirects) for url in chunk]
                
                # Process tasks
                for task in asyncio.as_completed(tasks):
                    result = await task
                    url = result['url']
                    self.processed_urls.add(url)
                    
                    # Track error by type for diagnostics
                    if 'error' in result:
                        error_type = result.get('error', '').split(':', 1)[0]
                        if not error_type:
                            error_type = "Unknown"
                        
                        if error_type not in results['error_stats']:
                            results['error_stats'][error_type] = 0
                        results['error_stats'][error_type] += 1
                    
                    # Handle failed URLs
                    if 'error' in result and 'html' not in result:
                        error_entry = {
                            'url': url,
                            'error': result['error'],
                            'status': result.get('status'),
                            'timestamp': result['timestamp']
                        }
                        
                        results['failed'].append(error_entry)
                        
                        # Save detailed error info
                        error_filename = self.get_safe_filename(url)
                        error_file = os.path.join(self.error_dir, f"{error_filename}_error.json")
                        with open(error_file, 'w', encoding='utf-8') as f:
                            json.dump(error_entry, f, indent=2)
                        
                        logger.warning(f"Failed to fetch: {url} - {result['error']}")
                        continue
                    
                    # Process HTML content
                    if 'html' in result:
                        # Extract text content
                        text = self.extract_text_from_html(result['html'], url)
                        
                        # Skip pages with insufficient content
                        if not text or len(text) < self.min_text_length:
                            error_entry = {
                                'url': url,
                                'error': f'Insufficient content (length: {len(text)})',
                                'status': result.get('status'),
                                'timestamp': result['timestamp']
                            }
                            
                            results['failed'].append(error_entry)
                            
                            # Track this error type
                            if 'Insufficient content' not in results['error_stats']:
                                results['error_stats']['Insufficient content'] = 0
                            results['error_stats']['Insufficient content'] += 1
                            
                            logger.warning(f"Insufficient content: {url} (length: {len(text)})")
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
                        
                        logger.info(f"Successfully processed: {url} (text length: {len(text)})")
                
                # Add a small delay between chunks to avoid overwhelming servers
                if i + chunk_size < len(unique_urls):
                    await asyncio.sleep(2 * random.uniform(0.8, 1.2))
        
        # Save overall results
        results['end_time'] = datetime.now().isoformat()
        results['total_urls'] = len(unique_urls)
        results['successful_count'] = len(results['successful'])
        results['failed_count'] = len(results['failed'])
        results['error_counts'] = self.error_counts
        
        # Calculate success rate
        if len(unique_urls) > 0:
            results['success_rate'] = results['successful_count'] / len(unique_urls) * 100
        else:
            results['success_rate'] = 0
        
        # Generate error summary
        error_summary = os.path.join(self.output_dir, 'error_summary.json')
        with open(error_summary, 'w', encoding='utf-8') as f:
            json.dump({
                'error_counts': self.error_counts,
                'error_stats': results['error_stats'],
                'success_rate': results['success_rate'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Generate extraction summary
        summary_path = os.path.join(self.output_dir, 'extraction_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in results.items() if k != 'documents'}, f, indent=2)
        
        logger.info(f"Extraction complete: {results['successful_count']} successful ({results['success_rate']:.1f}%), {results['failed_count']} failed")
        logger.info(f"Error summary: {results['error_stats']}")
        
        return results