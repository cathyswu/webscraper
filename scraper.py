import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://pytorch.org/docs/stable/index.html",
        )
        
        # Save the markdown content to a file
        with open("pytorch_docs.md", "w", encoding="utf-8") as f:
            f.write(result.markdown)
        
        print("PyTorch documentation saved to pytorch_docs.md")

if __name__ == "__main__":
    asyncio.run(main())

    import asyncio
from crawl4ai import *