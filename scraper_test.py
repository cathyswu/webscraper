import asyncio
from crawl4ai import AsyncWebCrawler, BestFirstCrawlingStrategy, CrawlerRunConfig, DomainFilter, FilterChain, KeywordRelevanceScorer, LXMLWebScrapingStrategy, URLPatternFilter, ContentTypeFilter
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import json
import time

async def main():
    content_dict = {}
    start_time = time.time()

    filter_chain = FilterChain([
        DomainFilter(
            allowed_domains=["docs.pytorch.org"],
            blocked_domains=["discuss.pytorch.org"]
        ),

        # # Target specific PyTorch documentation patterns
        # URLPatternFilter(patterns=[
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
        # ]),

        ContentTypeFilter(allowed_types=["text/html"])
    ])

    prune_filter = PruningContentFilter(
        # Lower → more content retained, higher → more content pruned
        threshold=0.4,           
        # "fixed" or "dynamic"
        threshold_type="dynamic",  
        # Ignore nodes with <5 words
        min_word_threshold=30      
    )

    md_generator = DefaultMarkdownGenerator(content_filter=prune_filter)

    # Use strong, domain-specific keywords to improve relevance
    scorer = KeywordRelevanceScorer(
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
        weight=0.7
    )

    strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        url_scorer=scorer,
        max_pages=200,
        filter_chain = filter_chain
    )

    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        # Content filtering
        word_count_threshold=30,
        excluded_tags=[
            "form", "header", "footer", "nav", "aside", "script", "style"
        ],
        exclude_external_links=True,

        # Content processing
        process_iframes=False,
        remove_overlay_elements=True,
        
        
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        #=["sphinxsidebar", "related", "headerlinks", "nav"],
        #excluded_ids=["top", "bottom", "searchbox"]
        
        # Cache control
        # cache_mode=CacheMode.ENABLED  # Use cache if available
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url="https://pytorch.org/docs/stable/",
            config=run_config
        )
        
        for result in results:
            if result.success:
                url = result.url
                
                # Skip malformed URLs with .html/ pattern
                if '.html/' in url:
                    print(f"⚠️  Skipped malformed URL: {url}")
                    continue
                
                markdown = result.markdown.fit_markdown
                
                # Skip pages with error messages
                if "does not contain the requested file" in markdown:
                    print(f"Skipped error page: {url}")
                    continue
                
                if "The site configured at this address" in markdown:
                    print(f"Skipped site error page: {url}")
                    continue
                
                # Only add valid content
                content_dict[url] = markdown
                print(f"Added: {url}")
                
            else:
                print(f"Crawl failed: {result.error_message}")
                print(f"Status code: {result.status_code}")
        
        print(f"\nTotal valid pages collected: {len(content_dict)}")
        
        with open("pytorch_docs.json", "w", encoding="utf-8") as f:
            json.dump(content_dict, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()  # ⏱️ end timing
        elapsed_time = end_time - start_time
        print(f"\nScraping completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())