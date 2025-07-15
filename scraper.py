import asyncio
from crawl4ai import *

async def main():
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url="https://docs.pytorch.org/docs/stable/library.html",
        )
        
        # Save the markdown content to a file
        with open("pytorch_docs_library.md", "w", encoding="utf-8") as f:
            f.write(result.markdown)
        
        print("PyTorch documentation saved to pytorch_docs_library.md")

if __name__ == "__main__":
    asyncio.run(main())

    import asyncio
from crawl4ai import *