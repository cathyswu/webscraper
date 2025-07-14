import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import os
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import logging

class PyTorchDocScraper:
    def __init__(self, base_url: str = "https://pytorch.org/docs/stable/", 
                 delay: float = 1.0, max_pages: int = 100):
        """
        Initialize the PyTorch documentation scraper.
        
        Args:
            base_url: Base URL for PyTorch documentation
            delay: Delay between requests (seconds) to be respectful
            max_pages: Maximum number of pages to scrape
        """
        self.base_url = base_url
        self.delay = delay
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_page_content(self, url: str) -> Dict:
        """
        Extract content from a single page.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary containing page data
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No Title"
            
            # Extract main content (adjust selectors based on PyTorch docs structure)
            content_selectors = [
                'article',
                '.rst-content',
                '.document',
                'main',
                '#main-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove navigation, sidebar, and other non-content elements
                    for elem in content_elem.select('.toctree-wrapper, .sidebar, nav, header, footer'):
                        elem.decompose()
                    content_text = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # Extract code blocks
            code_blocks = []
            for code_elem in soup.select('pre code, .highlight pre'):
                code_blocks.append(code_elem.get_text().strip())
            
            # Extract headings structure
            headings = []
            for heading in soup.select('h1, h2, h3, h4, h5, h6'):
                headings.append({
                    'level': heading.name,
                    'text': heading.get_text().strip()
                })
            
            # Extract links to other documentation pages
            doc_links = []
            for link in soup.select('a[href]'):
                href = link.get('href')
                if href:
                    full_url = urljoin(url, href)
                    if self.is_doc_url(full_url):
                        doc_links.append(full_url)
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'code_blocks': code_blocks,
                'headings': headings,
                'doc_links': doc_links,
                'content_length': len(content_text),
                'code_block_count': len(code_blocks),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': "Error",
                'content': "",
                'code_blocks': [],
                'headings': [],
                'doc_links': [],
                'content_length': 0,
                'code_block_count': 0,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def is_doc_url(self, url: str) -> bool:
        """
        Check if URL is a documentation page we want to scrape.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL should be scraped
        """
        parsed = urlparse(url)
        
        # Only scrape pytorch.org documentation
        if parsed.netloc != 'pytorch.org':
            return False
        
        # Skip certain types of pages
        skip_patterns = [
            '/docs/stable/_static/',
            '/docs/stable/_sources/',
            '.pdf',
            '.zip',
            '#',  # Skip anchor links
            'javascript:',
            'mailto:'
        ]
        
        for pattern in skip_patterns:
            if pattern in url:
                return False
        
        # Include documentation pages
        return '/docs/' in url
    
    def discover_urls(self, start_url: str) -> List[str]:
        """
        Discover documentation URLs starting from a base URL.
        
        Args:
            start_url: Starting URL for discovery
            
        Returns:
            List of discovered URLs
        """
        urls_to_visit = [start_url]
        discovered_urls = set()
        
        while urls_to_visit and len(discovered_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in discovered_urls:
                continue
                
            discovered_urls.add(current_url)
            self.logger.info(f"Discovering URLs from: {current_url}")
            
            page_data = self.get_page_content(current_url)
            
            # Add new URLs to visit
            for link in page_data.get('doc_links', []):
                if link not in discovered_urls and len(discovered_urls) < self.max_pages:
                    urls_to_visit.append(link)
            
            time.sleep(self.delay)
        
        return list(discovered_urls)
    
    def scrape_documentation(self, start_urls: List[str] = None) -> List[Dict]:
        """
        Scrape PyTorch documentation.
        
        Args:
            start_urls: List of starting URLs. If None, uses default starting points.
            
        Returns:
            List of scraped page data
        """
        if start_urls is None:
            start_urls = [
                self.base_url,
                "https://pytorch.org/docs/stable/torch.html",
                "https://pytorch.org/docs/stable/nn.html",
                "https://pytorch.org/docs/stable/optim.html",
                "https://pytorch.org/docs/stable/torchvision/index.html"
            ]
        
        # Discover all URLs to scrape
        all_urls = set()
        for start_url in start_urls:
            discovered = self.discover_urls(start_url)
            all_urls.update(discovered)
        
        all_urls = list(all_urls)[:self.max_pages]
        
        self.logger.info(f"Starting to scrape {len(all_urls)} URLs")
        
        # Scrape each URL
        for i, url in enumerate(all_urls, 1):
            if url in self.visited_urls:
                continue
                
            self.logger.info(f"Scraping {i}/{len(all_urls)}: {url}")
            
            page_data = self.get_page_content(url)
            self.scraped_data.append(page_data)
            self.visited_urls.add(url)
            
            time.sleep(self.delay)
        
        return self.scraped_data
    
    def save_to_json(self, filename: str = "pytorch_docs_dataset.json"):
        """Save scraped data to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Data saved to {filename}")
    
    def save_to_csv(self, filename: str = "pytorch_docs_dataset.csv"):
        """Save scraped data to CSV file."""
        if not self.scraped_data:
            self.logger.warning("No data to save")
            return
        
        # Flatten the data for CSV
        flattened_data = []
        for item in self.scraped_data:
            flattened_item = {
                'url': item['url'],
                'title': item['title'],
                'content': item['content'],
                'content_length': item['content_length'],
                'code_block_count': item['code_block_count'],
                'headings_count': len(item['headings']),
                'code_blocks': '\n---\n'.join(item['code_blocks']),
                'timestamp': item['timestamp']
            }
            if 'error' in item:
                flattened_item['error'] = item['error']
            flattened_data.append(flattened_item)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
        self.logger.info(f"Data saved to {filename}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the scraped data."""
        if not self.scraped_data:
            return {}
        
        total_pages = len(self.scraped_data)
        total_content_length = sum(item['content_length'] for item in self.scraped_data)
        total_code_blocks = sum(item['code_block_count'] for item in self.scraped_data)
        
        return {
            'total_pages': total_pages,
            'total_content_length': total_content_length,
            'average_content_length': total_content_length / total_pages if total_pages > 0 else 0,
            'total_code_blocks': total_code_blocks,
            'average_code_blocks_per_page': total_code_blocks / total_pages if total_pages > 0 else 0
        }

if __name__ == "__main__":
    # Initialize scraper
    scraper = PyTorchDocScraper(
        delay=1.0,  # 1 second between requests
        max_pages=50  # Limit for testing
    )
    
    print("Starting PyTorch documentation scraping...")
    data = scraper.scrape_documentation()
    
    scraper.save_to_json()
    scraper.save_to_csv()
    
    stats = scraper.get_statistics()
    print(f"\nScraping completed!")
    print(f"Total pages scraped: {stats['total_pages']}")
    print(f"Total content length: {stats['total_content_length']:,} characters")
    print(f"Average content length: {stats['average_content_length']:.0f} characters per page")
    print(f"Total code blocks: {stats['total_code_blocks']}")
    print(f"Average code blocks per page: {stats['average_code_blocks_per_page']:.1f}")