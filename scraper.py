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

def build_filter_chain(allowed_domains, blocked_domains=None, url_patterns=None):
    filters = [DomainFilter(allowed_domains=allowed_domains, blocked_domains=blocked_domains or [])]
    
    if url_patterns:
        filters.append(URLPatternFilter(patterns=url_patterns))
    
    filters.append(ContentTypeFilter(allowed_types=["text/html"]))
    return FilterChain(filters)

def build_run_config(filter_chain, keywords, output_threshold=30):
    scorer = KeywordRelevanceScorer(keywords=keywords, weight=0.7)

    strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        url_scorer=scorer,
        max_pages=200,
        filter_chain=filter_chain
    )

    prune_filter = PruningContentFilter(
        threshold=0.4,
        threshold_type="dynamic",
        min_word_threshold=30
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    return CrawlerRunConfig(
        markdown_generator=md_generator,
        word_count_threshold=output_threshold,
        excluded_tags=["form", "header", "footer", "nav", "aside", "script", "style"],
        exclude_external_links=True,
        process_iframes=False,
        remove_overlay_elements=True,
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

# === RESULT HANDLER ===

def process_results(results):
    content_dict = {}

    for result in results:
        for link in result.links["internal"]:
                print(f"Internal link: {link['href']}")

        if result.success:
            url = result.url
            if '.html/' in url:
                print(f"Skipped malformed URL: {url}")
                continue

            markdown = result.markdown.fit_markdown
            if any(x in markdown for x in ["does not contain the requested file", "The site configured at this address"]):
                print(f"Skipped error page: {url}")
                continue

            content_dict[url] = markdown
            print(f"Added: {url}")
        else:
            print(f"Crawl failed: {result.error_message}")
            print(f"Status code: {result.status_code}")

    return content_dict

# === MAIN CRAWLER ===

async def crawl_site(start_url, allowed_domains, keywords, url_patterns=None, blocked_domains=None, output_file="output.json"):
    start_time = time.time()

    filter_chain = build_filter_chain(allowed_domains, blocked_domains, url_patterns)
    run_config = build_run_config(filter_chain, keywords)

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url=start_url, config=run_config)

        content = process_results(results)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"\nScraping completed in {time.time() - start_time:.2f} seconds.")
    print(f"Total valid pages: {len(content)}")

# MAIN USAGE

if __name__ == "__main__":
    asyncio.run(crawl_site(
        start_url="https://pytorch.org/docs/stable/",
        allowed_domains=["docs.pytorch.org"],
        blocked_domains=["discuss.pytorch.org"],
        keywords=[
            # Core PyTorch concepts
            "pytorch", "tensor", "torch.nn", "autograd", "gradient",
            "neural network", "deep learning", "model", "layer",
            
            # Common operations
            "forward", "backward", "loss", "optimizer", "training",
            "inference", "module", "parameter", "requires_grad",
            
            # Data types and operations
            "float32", "cuda", "device", "dtype", "reshape", "view",
            "matmul", "conv2d", "linear", "relu", "softmax",
            
            # Advanced concepts
            "distributed", "jit", "torchscript", "onnx", "quantization"
        ],
        # url_patterns=[
        #     r".*/torch\.nn\..*",           # Neural network modules
        #     r".*/torch\.tensor\..*",       # Tensor operations
        #     r".*/torch\.autograd\..*",     # Automatic differentiation
        #     r".*/torch\.optim\..*",        # Optimizers
        #     r".*/torch\.utils\.data\..*",  # Data utilities
        #     r".*/generated/torch\..*",     # Generated API docs
        #     r".*/tutorials/.*",            # Tutorials
        #     r".*/notes/.*",               # Technical notes
        #     # Exclude less relevant sections
        #     r"^(?!.*/(community|audio|text|vision|mobile|quantization)/)",
        # ],
        output_file="pytorch_docs.json"
    ))

    
    asyncio.run(crawl_site(
        start_url="https://www.usgs.gov/science",
        allowed_domains=["usgs.gov"],
        blocked_domains=["usgs.gov/staff-profiles"],
        keywords=[
            # General geospatial and scientific terms
            "geospatial", "remote sensing", "satellite", "imagery", "lidar",
            "elevation", "raster", "vector", "projection", "coordinate",
            "earth science", "geology", "mapping", "topography", "GIS",
            "hydrology", "land cover", "climate", "terrain", "spatial analysis",
            "geographic", "usgs", "national map", "3DEP", "dem", "geoid",
            "earthquake", "flood", "soil", "vegetation"
        ],
        # url_patterns=[
        #     r".*/science-explorer/.*",             # datasets and tools
        #     # r".*/maps/.*",             # mapping applications
        #     # r".*/remote-sensing.*",    # remote sensing pages
        #     # r".*/core-science-systems.*",  # org-level technical content
        #     # r".*/faqs/.*",             # scientific FAQ pages
        #     # r".*/earthquake.*",        # hazard-related content
        # ],
        output_file="usgs_science_docs.json"
    ))


