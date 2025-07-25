import asyncio
import json
from typing import Dict, List
from crawl4ai import AsyncWebCrawler, BestFirstCrawlingStrategy, CrawlerRunConfig, DomainFilter, FilterChain, LXMLWebScrapingStrategy, ContentTypeFilter
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.deep_crawling.filters import SEOFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

def create_filter_chain(allowed_domains: List[str], blocked_domains: List[str] = None, seo_keywords: List[str] = None) -> FilterChain:
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

    if seo_keywords:
        filters.append(SEOFilter(threshold=0.5, keywords=seo_keywords))

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
                     excluded_tags: List[str] = None, prune_threshold: float = 0.4, seo_keywords: List[str] = None) -> CrawlerRunConfig:
    """Create crawler run configuration"""
    if excluded_tags is None:
        excluded_tags = ["form", "header", "footer", "nav", "aside", "script", "style"]
    
    md_generator = create_markdown_generator(prune_threshold)
    strategy = create_crawling_strategy(max_depth, max_pages, allowed_domains, blocked_domains, seo_keywords)
    
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
    
    if len(markdown) < 10:
            print(f"❗Skipped empty/short content ({len(markdown)} chars): {url}")
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
                       prune_threshold: float = 0.4, seo_keywords: List[str] = None, output_file: str = "crawl_results.json") -> Dict[str, str]:
    
    
    """Main crawling function"""
    if blocked_domains is None:
        blocked_domains = []
    if skip_patterns is None:
        skip_patterns = []
    if seo_keywords is None:
        seo_keywords = []
    
    print(f"Starting crawl for: {start_url}")
    
    browser_config = BrowserConfig(verbose=True)
    run_config = create_run_config(
        max_depth=max_depth,
        max_pages=max_pages,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
        css_selector=css_selector,
        excluded_tags=excluded_tags,
        prune_threshold=prune_threshold,
        seo_keywords=seo_keywords
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
        "seo_keywords": 
            ["GIS", "remote sensing", "geospatial analysis", "spatial data", "topography",
            "satellite imagery", "geographic information system", "elevation models",
            "coordinate systems", "georeferencing", "spatial resolution", "land cover",
            "cartography", "shapefile", "geodatabase", "map projection", "terrain analysis"],
        "output_file": "usgs_geospatial.json"
    }

    python_org_config = {
        "start_url": "https://docs.python.org/3/",
        "allowed_domains": ["python.org", "docs.python.org"],
        "max_depth": 3,
        "max_pages": 300,
        "css_selector": ".main-content, .document, .section, [role='main'], .body",
        "output_file": "python_org_docs.json"
    }
    
    typescript_config = {
        "start_url": "https://www.typescriptlang.org/docs/",
        "allowed_domains": ["typescriptlang.org"],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": ".handbook-content, [role='main']",
        "output_file": "typescript_docs.json"
    }

    go_config = {
        "start_url": "https://go.dev/doc/",
        "allowed_domains": ["go.dev"],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": "#main-content, #content",
        "output_file": "go_docs.json"
    }

    django_config = {
        "start_url": "https://docs.djangoproject.com/en/5.2/",
        "allowed_domains": ["docs.djangoproject.com"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": "article#docs-content, #docs-content, main, #main-content, article",
        "output_file": "django_docs.json"
    }

    flask_config = {
        "start_url": "https://flask.palletsprojects.com/en/stable/",
        "allowed_domains": ["flask.palletsprojects.com"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": ".body, [role='main']",
        "output_file": "flask_docs.json"
    }

    java_config = {
        "start_url": "https://docs.oracle.com/en/java/javase/24/docs/api/index.html",
        "allowed_domains": ["docs.oracle.com"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": "[role='main']",
        "output_file": "java_docs.json"
    }

    node_config = {
        "start_url": "https://nodejs.org/docs/latest/api/",
        "allowed_domains": ["nodejs.org"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": "main, [role='main'], #apicontent",
        "output_file": "node_docs.json"
    }

    react_config = {
        "start_url": "https://react.dev/reference/react",
        "allowed_domains": ["react.dev"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": ".min-w-0.isolate",
        "output_file": "react_docs.json"
    }

    angular_config = {
        "start_url": "https://angular.dev/overview",
        "allowed_domains": ["angular.dev"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 300,
        "css_selector": "docs-docs, docs-viewer, .docs-viewer, .docs-with-TOC",
        "skip_patterns": ["If you think this is a mistake"],
        "output_file": "angular_docs.json"
    }

    numpy_config = {
        "start_url": "https://numpy.org/doc/stable/",
        "allowed_domains": ["numpy.org"],
        "blocked_domains": [],
        "max_depth": 5,
        "max_pages": 400,
        "css_selector": "#main-content, .main.bd-main, [role='main']",
        "output_file": "numpy_docs.json"
    }

    opencv_config = {
        "start_url": "https://docs.opencv.org/4.x/",
        "allowed_domains": ["docs.opencv.org"],
        "blocked_domains": [],
        "max_depth": 5,
        "max_pages": 400,
        "css_selector": ".textblock, .contents",
        "output_file": "opencv_docs.json"
    }

    javascript_config = {
        "start_url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "allowed_domains": ["developer.mozilla.org"],
        "blocked_domains": [],
        "max_depth": 3,
        "max_pages": 300,
        "css_selector": ".main-page-content",
        "output_file": "javascript_docs.json"
    }

    jsdoc_config = {
        "start_url": "https://jsdoc.app/",
        "allowed_domains": ["jsdoc.app"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 400,
        "css_selector": "article",
        "output_file": "jsdoc_docs.json"
    }

    material_angular_config = {
        "start_url": "https://material.angular.dev/guides",
        "allowed_domains": ["material.angular.dev"],
        "blocked_domains": [],
        "max_depth": 4,
        "max_pages": 400,
        "css_selector": "main, .docs-markdown",
        "output_file": "material_angular_docs.json"
    }

    cuda_config = {
        "start_url": "https://docs.nvidia.com/cuda/",
        "allowed_domains": ["docs.nvidia.com"],
        "blocked_domains": [],
        "max_depth": 1,
        "max_pages": 50,
        "css_selector": "[role='main'], .document",
        "output_file": "cuda_docs.json"
    }

    defensive_programming_config = {
        "start_url": "https://en.wikipedia.org/wiki/Defensive_programming",
        "allowed_domains": ["en.wikipedia.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "#content",
        "seo_keywords": [
            "input validation",
            "error handling",
            "exception handling",
            "assertions",
            "fail-safe defaults",
            "sanitization",
            "fail fast"
            "guard clauses",
            "code contracts",
            "redundancy",
            "exception safety",
            "preconditions",
            "postconditions",
            "defensive copying",
            "null checks",
            "immutable objects",
            "graceful degradation",
            "boundary checks",
            "secure coding",
            "fault tolerance",
            "robustness",
        ],
        "output_file": "defensive_programming.json"
    }

    rust_config = {
        "start_url": "https://doc.rust-lang.org/stable/",
        "allowed_domains": ["doc.rust-lang.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 400,
        "css_selector": ".rustdoc, #main-content, #content, .content, main",
        "output_file": "rust_docs.json"
    }

    vue_config = {
        "start_url": "https://vuejs.org/api/",
        "allowed_domains": ["vuejs.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "main, .content",
        "output_file": "vue_docs.json"
    }

    springboot_config = {
        "start_url": "https://docs.spring.io/spring-boot/documentation.html",
        "allowed_domains": ["docs.spring.io"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": ".doc, [role='main'], main",
        "output_file": "springboot_docs.json"
    }

    dotnet_config = {
        "start_url": "https://learn.microsoft.com/en-us/dotnet/framework/",
        "allowed_domains": ["learn.microsoft.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "#main, main, [role='main']",
        "output_file": "springboot_docs.json"
    }

    restapi_config = {
        "start_url": "https://docs.github.com/en/rest?apiVersion=2022-11-28",
        "allowed_domains": ["docs.github.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "#main-content, main",
        "output_file": "restapi_docs.json"
    }

    html_config = {
        "start_url": "https://developer.mozilla.org/en-US/docs/Web/HTML",
        "allowed_domains": ["developer.mozilla.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": ".main-content, main, #content",
        "output_file": "html_docs.json"
    }

    git_config = {
        "start_url": "https://git-scm.com/docs",
        "allowed_domains": ["git-scm.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "#main",
        "output_file": "git_docs.json"
    }

    docker_config = {
        "start_url": "https://docs.docker.com/reference/",
        "allowed_domains": ["docs.docker.com"],
        "blocked_domains": [],
        "max_depth": 3,
        "max_pages": 400,
        "css_selector": "article",
        "output_file": "docker_docs.json"
    }

    kubernetes_config = {
        "start_url": "https://kubernetes.io/docs/home/",
        "allowed_domains": ["kubernetes.io"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 200,
        "css_selector": "main, [role='main']",
        "output_file": "kubernetes_docs.json"
    }

    terraform_config = {
        "start_url": "https://developer.hashicorp.com/terraform/docs",
        "allowed_domains": ["developer.hashicorp.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "main, #main",
        "output_file": "terraform_docs.json"
    }

    ansible_config = {
        "start_url": "https://docs.ansible.com/ansible/latest/index.html",
        "allowed_domains": ["docs.ansible.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "[itemprop='articleBody']",
        "output_file": "ansible_docs.json"
    }

    jenkins_config = {
        "start_url": "https://www.jenkins.io/doc/book/getting-started/",
        "allowed_domains": ["www.jenkins.io"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": ".ctc, .col-lg-9",
        "output_file": "jenkins_docs.json"
    }

    grafana_config = {
        "start_url": "https://grafana.com/docs/",
        "allowed_domains": ["grafana.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": ".main-content, main, #main",
        "output_file": "grafana_docs.json"
    }

    nginx_config = {
        "start_url": "https://nginx.org/en/docs/",
        "allowed_domains": ["nginx.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "#content",
        "output_file": "nginx_docs.json"
    }

    postgresql_config = {
        "start_url": "https://www.postgresql.org/docs/current/index.html",
        "allowed_domains": ["www.postgresql.org"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "#docContent",
        "output_file": "postgresql_docs.json"
    }

    mysql_config = {
        "start_url": "https://dev.mysql.com/doc/refman/8.4/en/introduction.html",
        "allowed_domains": ["dev.mysql.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "#docs-main-inner",
        "output_file": "mysql_docs.json"
    }

    mongodb_config = {
        "start_url": "https://www.mongodb.com/docs/manual/",
        "allowed_domains": ["www.mongodb.com"],
        "blocked_domains": [],
        "max_depth": 2,
        "max_pages": 300,
        "css_selector": "#template-container",
        "output_file": "mongodb_docs.json"
    }

    selected_config = mongodb_config
    
    content_dict = await crawl_website(**selected_config)
    
    print(f"Crawling completed! Found {len(content_dict)} pages.")

if __name__ == "__main__":
    asyncio.run(main())