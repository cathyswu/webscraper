import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

async def main():
    # Use strong, domain-specific keywords to improve relevance
    scorer = KeywordRelevanceScorer(
        keywords=["pytorch", "tensor", "model", "neural network", "torch.nn", "autograd"],
        weight=0.5  # Strong emphasis on keyword matches
    )

    strategy = BestFirstCrawlingStrategy(
        max_depth=2,
        include_external=False,
        url_scorer=scorer,
        max_pages=50
    )

    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True
    )

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(
            url="https://pytorch.org/docs/stable/index.html",
            config=config
        )

        print(f"Crawled {len(results)} pages in total")

if __name__ == "__main__":
    asyncio.run(main())
