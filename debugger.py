'''
Use this to quickly preview or extract the raw HTML or Markdown content of a single page.
'''

import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    # Create an instance of AsyncWebCrawler
    async with AsyncWebCrawler() as crawler:
        # Run the crawler on a URL
        result = await crawler.arun(url="https://www.jenkins.io/doc/")

        # Print the extracted content (result.html, result.markdown)
        print(result.html)

# Run the async main function
asyncio.run(main())