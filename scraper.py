import asyncio
from crawl4ai import AsyncWebCrawler, BestFirstCrawlingStrategy, CrawlerRunConfig, DomainFilter, FilterChain, KeywordRelevanceScorer, LXMLWebScrapingStrategy, URLPatternFilter, ContentTypeFilter
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import json
from typing import Dict, List, Optional

def create_filter_chain(allowed_domains: List[str], blocked_domains: List[str] = None) -> FilterChain:
    """Create filter chain with domain and content type filters"""
    if blocked_domains is None:
        blocked_domains = []
    
    filters = [
        DomainFilter(
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains
        ),
        ContentTypeFilter(allowed_types=["text/html"])
    ]
    return FilterChain(filters)

def create_markdown_generator(prune_threshold: float = 0.4) -> DefaultMarkdownGenerator:
    """Create markdown generator with content pruning"""
    prune_filter = PruningContentFilter(
        threshold=prune_threshold,
        threshold_type="dynamic"
    )
    return DefaultMarkdownGenerator(content_filter=prune_filter)

def create_crawling_strategy(max_depth: int, max_pages: int, allowed_domains: List[str], 
                           blocked_domains: List[str] = None, include_external: bool = False) -> BestFirstCrawlingStrategy:
    """Create crawling strategy with specified parameters"""
    filter_chain = create_filter_chain(allowed_domains, blocked_domains)
    
    return BestFirstCrawlingStrategy(
        max_depth=max_depth,
        include_external=include_external,
        max_pages=max_pages,
        filter_chain=filter_chain
    )

def create_run_config(max_depth: int, max_pages: int, allowed_domains: List[str], 
                     blocked_domains: List[str] = None, css_selector: str = None,
                     excluded_tags: List[str] = None, prune_threshold: float = 0.4) -> CrawlerRunConfig:
    """Create crawler run configuration"""
    if excluded_tags is None:
        excluded_tags = ["form", "header", "footer", "nav", "aside", "script", "style"]
    
    md_generator = create_markdown_generator(prune_threshold)
    strategy = create_crawling_strategy(max_depth, max_pages, allowed_domains, blocked_domains)
    
    config_params = {
        'markdown_generator': md_generator,
        'excluded_tags': excluded_tags,
        'exclude_external_links': True,
        'process_iframes': False,
        'remove_overlay_elements': True,
        'deep_crawl_strategy': strategy,
        'scraping_strategy': LXMLWebScrapingStrategy(),
        'verbose': True,
    }
    
    if css_selector:
        config_params['css_selector'] = css_selector
        
    return CrawlerRunConfig(**config_params)

def should_skip_content(markdown: str, url: str, skip_patterns: List[str]) -> bool:
    """Check if content should be skipped based on patterns"""
    if '.html/' in url:
        print(f"Skipped malformed URL: {url}")
        return True
    
    for pattern in skip_patterns:
        if pattern in markdown:
            print(f"Skipped content with pattern '{pattern}': {url}")
            return True
    
    return False

def process_results(results, skip_patterns: List[str]) -> Dict[str, str]:
    """Process crawl results and return content dictionary"""
    content_dict = {}
    
    for result in results:
        if result.success:
            url = result.url
            markdown = result.markdown.fit_markdown
            
            if should_skip_content(markdown, url, skip_patterns):
                continue
            
            content_dict[url] = markdown
            print(f"Added: {url}")
        else:
            print(f"Crawl failed: {result.error_message}")
            print(f"Status code: {result.status_code}")
    
    return content_dict

def save_results(content_dict: Dict[str, str], output_file: str):
    """Save results to JSON file"""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(content_dict, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

async def crawl_website(start_url: str, allowed_domains: List[str], blocked_domains: List[str] = None,
                       max_depth: int = 2, max_pages: int = 200, css_selector: str = None,
                       excluded_tags: List[str] = None, skip_patterns: List[str] = None,
                       prune_threshold: float = 0.4, output_file: str = "crawl_results.json") -> Dict[str, str]:
    
    
    """Main crawling function"""
    if blocked_domains is None:
        blocked_domains = []
    if skip_patterns is None:
        skip_patterns = []
    
    print(f"Starting crawl for: {start_url}")
    
    browser_config = BrowserConfig(verbose=True)
    run_config = create_run_config(
        max_depth=max_depth,
        max_pages=max_pages,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        css_selector=css_selector,
        excluded_tags=excluded_tags,
        prune_threshold=prune_threshold
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun(url=start_url, config=run_config)
        
        content_dict = process_results(results, skip_patterns)
        
        print(f"\nTotal valid pages collected: {len(content_dict)}")
        
        if content_dict:
            save_results(content_dict, output_file)
        
        return content_dict

async def main():
    pytorch_config = {
        "start_url": "https://docs.pytorch.org/docs/stable",
        "allowed_domains": ["docs.pytorch.org"],
        "blocked_domains": ["discuss.pytorch.org"],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": ".rst-content",
        "skip_patterns": [
            "does not contain the requested file",
            "The site configured at this address",
        ],
        "output_file": "pytorch_docs.json"
    }
    
    tensorflow_config = {
        "start_url": "https://www.tensorflow.org/tfx/tutorials",
        "allowed_domains": ["tensorflow.org"],
        "max_depth": 3,
        "max_pages": 200,
        "css_selector": ".devsite-article-body",
        "output_file": "tensorflow_docs.json"
    }

    usgs_config = {
        "start_url": "https://www.usgs.gov/science",
        "allowed_domains": ["usgs.gov"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 150,
        "css_selector": "#main-content", 
        "output_file": "usgs_geospatial.json"
    }
    
    selected_config = usgs_config
    
    content_dict = await crawl_website(**selected_config)
    
    print(f"Crawling completed! Found {len(content_dict)} pages.")

if __name__ == "__main__":
    asyncio.run(main())