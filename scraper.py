import asyncio
import time
import json
from crawl4ai import (
    AsyncWebCrawler, BestFirstCrawlingStrategy, CrawlerRunConfig,
    DomainFilter, FilterChain, KeywordRelevanceScorer,
    LXMLWebScrapingStrategy, URLPatternFilter, ContentTypeFilter
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_configs import BrowserConfig

# == CONFIG HELPERS ==

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["docs.pytorch.org"],
            blocked_domains=["discuss.pytorch.org"]
        ),
        ContentTypeFilter(allowed_types=["text/html"])
    ])

    prune_filter = PruningContentFilter(
        threshold=0.4,           
        threshold_type="dynamic",      
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        max_pages=200,
        filter_chain = filter_chain
    )

    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=[
            "form", "header", "footer", "nav", "aside", "script", "style"
        ],
        # Target the main content area and exclude navigation/footer elements
        css_selector=".rst-content",
        exclude_external_links=True,

        process_iframes=False,
        remove_overlay_elements=True,
        
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url="https://docs.pytorch.org/docs/stable",
            config=run_config
        )
        
        for result in results:
            if result.success:
                url = result.url
                
                if '.html/' in url:
                    print(f"Skipped malformed URL: {url}")
                    continue
                
                markdown = result.markdown.fit_markdown
                
                skip_patterns = [
                    "does not contain the requested file",
                    "The site configured at this address",
                ]
                
                if any(pattern in markdown for pattern in skip_patterns):
                    print(f"Skipped navigation page: {url}")
                    continue
                
                content_dict[url] = markdown
                print(f"Added: {url}")
                
            else:
                print(f"Crawl failed: {result.error_message}")
                print(f"Status code: {result.status_code}")
        
        print(f"\nTotal valid pages collected: {len(content_dict)}")
        
        with open("pytorch_docs.json", "w", encoding="utf-8") as f:
            json.dump(content_dict, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    asyncio.run(main())