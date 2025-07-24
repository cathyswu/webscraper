# Web Scraper

A comprehensive asynchronous web scraping tool built with [Crawl4AI](https://docs.crawl4ai.com/) that crawls documentation sites and extracts clean content in markdown format.

## Features

- **Asynchronous crawling** for high performance
- **Smart content filtering** with domain and content type filters
- **Markdown generation** with content pruning
- **Configurable crawling strategies** with depth and page limits
- **Multiple site configurations** for popular documentation sites
- **Content pattern filtering** to skip unwanted pages
- **JSON output** for easy data processing (e.g. RAG pipelines)

## Requirements

### System Requirements
- Python 3.8+

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd webscraper
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv crawler_env
   source crawler_env/bin/activate  # On Windows: crawler_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration Parameters

Each configuration supports the following parameters:

### Core Parameters
- `start_url` (str): Starting URL for the crawl
- `allowed_domains` (List[str]): Domains that are allowed to be crawled
- `blocked_domains` (List[str], optional): Domains to exclude from crawling
- `max_depth` (int): Maximum depth to crawl from the starting URL
- `max_pages` (int): Maximum number of pages to crawl
- `output_file` (str): Output JSON file name

### Content Filtering
- `css_selector` (str, optional): CSS selector to target specific content areas
- `excluded_tags` (List[str], optional): HTML tags to exclude from content
- `skip_patterns` (List[str], optional): Text patterns that trigger page skipping
- `prune_threshold` (float): Content pruning threshold (0.0-1.0)

## Running the Script

To run the script from terminal:

```bash
python scraper.py
```

> The script will default to crawling the selected config at the bottom (e.g. `cuda_config`). You can modify `selected_config` in `main()` to crawl a different site.

Example:
```python
selected_config = tensorflow_config
```

## Debugging Tips

- If you see a lot of skipped pages or empty outputs, adjust the `css_selector`.
- Increase `max_pages` or `max_depth` if you're not getting enough content.
- Use `prune_threshold=0.0` to disable content pruning temporarily.