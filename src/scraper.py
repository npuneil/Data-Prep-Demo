"""
Web Scraper Module - Captures and parses public web pages for financial data.

Handles fetching, parsing, table extraction, and text normalization
from public financial web pages (SEC EDGAR, fund fact sheets, etc.).
"""

import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup, Tag
import pandas as pd


@dataclass
class ScrapedPage:
    """Represents a scraped web page with metadata."""
    url: str
    title: str = ""
    text_content: str = ""
    html_content: str = ""
    tables: List[pd.DataFrame] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fetch_timestamp: str = ""
    content_hash: str = ""
    status_code: int = 0
    fetch_duration_ms: float = 0
    word_count: int = 0
    table_count: int = 0


# Embedded sample financial data for offline demo capability
SAMPLE_FINANCIAL_HTML = """
<html>
<head><title>Fund Performance Summary - Q4 2025 Report</title></head>
<body>
<h1>Vanguard Investment Fund Performance Report</h1>
<p>Report Date: December 31, 2025 | Fund Family: Vanguard Group</p>

<h2>Vanguard 500 Index Fund Admiral Shares (VFIAX)</h2>
<p>ISIN: US9229087690 | CUSIP: 922908769 | Inception Date: November 13, 2000</p>
<p>Category: Large Blend | Style: Large Growth</p>
<p>Net Asset Value (NAV): $502.87 as of 12/31/2025</p>
<p>Total Net Assets: $442.8 Billion | Expense Ratio: 0.04% | SEC Yield: 1.28%</p>
<p>Rating: ★★★★★ (5 stars) | Analyst Rating: Gold</p>

<h3>Performance Returns</h3>
<table border="1">
<thead>
<tr><th>Period</th><th>Fund Return</th><th>Benchmark Return</th><th>+/- Benchmark</th></tr>
</thead>
<tbody>
<tr><td>1 Month</td><td>2.41%</td><td>2.43%</td><td>-0.02%</td></tr>
<tr><td>3 Month</td><td>5.87%</td><td>5.89%</td><td>-0.02%</td></tr>
<tr><td>YTD</td><td>26.29%</td><td>26.33%</td><td>-0.04%</td></tr>
<tr><td>1 Year</td><td>26.29%</td><td>26.33%</td><td>-0.04%</td></tr>
<tr><td>3 Year (Ann.)</td><td>9.47%</td><td>9.51%</td><td>-0.04%</td></tr>
<tr><td>5 Year (Ann.)</td><td>14.53%</td><td>14.57%</td><td>-0.04%</td></tr>
<tr><td>10 Year (Ann.)</td><td>12.86%</td><td>12.90%</td><td>-0.04%</td></tr>
</tbody>
</table>

<h3>Risk Metrics (3-Year)</h3>
<table border="1">
<thead>
<tr><th>Metric</th><th>Fund</th><th>Category Avg</th></tr>
</thead>
<tbody>
<tr><td>Standard Deviation</td><td>17.24</td><td>17.85</td></tr>
<tr><td>Sharpe Ratio</td><td>0.42</td><td>0.38</td></tr>
<tr><td>Beta</td><td>1.00</td><td>0.98</td></tr>
<tr><td>Alpha</td><td>-0.04</td><td>-0.63</td></tr>
<tr><td>R-Squared</td><td>100.00</td><td>96.42</td></tr>
</tbody>
</table>

<h3>Top 10 Holdings</h3>
<table border="1">
<thead>
<tr><th>#</th><th>Holding</th><th>Ticker</th><th>Sector</th><th>Weight (%)</th></tr>
</thead>
<tbody>
<tr><td>1</td><td>Apple Inc</td><td>AAPL</td><td>Technology</td><td>7.24</td></tr>
<tr><td>2</td><td>Microsoft Corp</td><td>MSFT</td><td>Technology</td><td>6.58</td></tr>
<tr><td>3</td><td>NVIDIA Corp</td><td>NVDA</td><td>Technology</td><td>6.12</td></tr>
<tr><td>4</td><td>Amazon.com Inc</td><td>AMZN</td><td>Consumer Cyclical</td><td>3.87</td></tr>
<tr><td>5</td><td>Alphabet Inc Class A</td><td>GOOGL</td><td>Communication Services</td><td>2.31</td></tr>
<tr><td>6</td><td>Meta Platforms Inc</td><td>META</td><td>Communication Services</td><td>2.51</td></tr>
<tr><td>7</td><td>Berkshire Hathaway Inc</td><td>BRK.B</td><td>Financial Services</td><td>1.82</td></tr>
<tr><td>8</td><td>Tesla Inc</td><td>TSLA</td><td>Consumer Cyclical</td><td>1.94</td></tr>
<tr><td>9</td><td>Broadcom Inc</td><td>AVGO</td><td>Technology</td><td>1.68</td></tr>
<tr><td>10</td><td>JPMorgan Chase & Co</td><td>JPM</td><td>Financial Services</td><td>1.42</td></tr>
</tbody>
</table>

<h3>Sector Allocation</h3>
<table border="1">
<thead>
<tr><th>Sector</th><th>Fund (%)</th><th>S&P 500 (%)</th></tr>
</thead>
<tbody>
<tr><td>Technology</td><td>31.42</td><td>31.42</td></tr>
<tr><td>Financial Services</td><td>13.18</td><td>13.18</td></tr>
<tr><td>Healthcare</td><td>11.67</td><td>11.67</td></tr>
<tr><td>Consumer Cyclical</td><td>10.53</td><td>10.53</td></tr>
<tr><td>Communication Services</td><td>9.12</td><td>9.12</td></tr>
<tr><td>Industrials</td><td>8.44</td><td>8.44</td></tr>
<tr><td>Consumer Defensive</td><td>5.87</td><td>5.87</td></tr>
<tr><td>Energy</td><td>3.56</td><td>3.56</td></tr>
<tr><td>Utilities</td><td>2.48</td><td>2.48</td></tr>
<tr><td>Real Estate</td><td>2.24</td><td>2.24</td></tr>
<tr><td>Basic Materials</td><td>1.49</td><td>1.49</td></tr>
</tbody>
</table>

<hr>

<h2>Vanguard Total Bond Market Index Fund Admiral (VBTLX)</h2>
<p>ISIN: US9219097683 | CUSIP: 921909768 | Inception Date: November 12, 2001</p>
<p>Category: Intermediate Core Bond | Base Currency: USD</p>
<p>Net Asset Value (NAV): $9.87 as of 12/31/2025</p>
<p>Total Net Assets: $108.4 Billion | Expense Ratio: 0.05% | SEC Yield: 4.52%</p>
<p>Rating: ★★★★ (4 stars) | Analyst Rating: Gold</p>

<h3>Performance Returns</h3>
<table border="1">
<thead>
<tr><th>Period</th><th>Fund Return</th><th>Benchmark Return</th><th>+/- Benchmark</th></tr>
</thead>
<tbody>
<tr><td>1 Month</td><td>-1.64%</td><td>-1.62%</td><td>-0.02%</td></tr>
<tr><td>3 Month</td><td>-3.06%</td><td>-3.04%</td><td>-0.02%</td></tr>
<tr><td>YTD</td><td>1.25%</td><td>1.28%</td><td>-0.03%</td></tr>
<tr><td>1 Year</td><td>1.25%</td><td>1.28%</td><td>-0.03%</td></tr>
<tr><td>3 Year (Ann.)</td><td>-2.41%</td><td>-2.38%</td><td>-0.03%</td></tr>
<tr><td>5 Year (Ann.)</td><td>-0.33%</td><td>-0.30%</td><td>-0.03%</td></tr>
<tr><td>10 Year (Ann.)</td><td>1.35%</td><td>1.38%</td><td>-0.03%</td></tr>
</tbody>
</table>

<h3>Risk Metrics (3-Year)</h3>
<table border="1">
<thead>
<tr><th>Metric</th><th>Fund</th><th>Category Avg</th></tr>
</thead>
<tbody>
<tr><td>Standard Deviation</td><td>7.14</td><td>7.42</td></tr>
<tr><td>Sharpe Ratio</td><td>-0.67</td><td>-0.72</td></tr>
<tr><td>Beta</td><td>1.00</td><td>0.95</td></tr>
<tr><td>Alpha</td><td>-0.03</td><td>-0.38</td></tr>
</tbody>
</table>

<footer>
<p>Data as of December 31, 2025. Past performance is no guarantee of future results.</p>
<p>Source: Vanguard Group | Report generated for demonstration purposes.</p>
</footer>
</body>
</html>
"""


class WebScraper:
    """Scrapes and parses public web pages for financial data extraction."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DataPrepAssistant/1.0 (Research Demo)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def fetch_page(self, url: str) -> ScrapedPage:
        """Fetch and parse a web page from URL."""
        page = ScrapedPage(url=url)
        page.fetch_timestamp = datetime.now(timezone.utc).isoformat()
        
        start_time = time.perf_counter()
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            page.status_code = response.status_code
            response.raise_for_status()
            
            page.html_content = response.text
            page.content_hash = hashlib.sha256(response.content).hexdigest()
            
        except requests.RequestException as e:
            page.metadata["error"] = str(e)
            page.metadata["fallback"] = True
            # Use sample data as fallback
            page.html_content = SAMPLE_FINANCIAL_HTML
            page.content_hash = hashlib.sha256(page.html_content.encode()).hexdigest()
            page.status_code = 200
            page.url = url or "demo://sample-fund-report"
        
        page.fetch_duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse the HTML
        self._parse_page(page)
        
        return page
    
    def fetch_sample(self) -> ScrapedPage:
        """Load the embedded sample financial data for demo/offline use."""
        page = ScrapedPage(url="demo://fund-performance-q4-2025")
        page.fetch_timestamp = datetime.now(timezone.utc).isoformat()
        page.html_content = SAMPLE_FINANCIAL_HTML
        page.content_hash = hashlib.sha256(page.html_content.encode()).hexdigest()
        page.status_code = 200
        page.metadata["source"] = "embedded_sample"
        page.metadata["description"] = "Vanguard Fund Performance Report Q4 2025 (Demo Data)"
        
        self._parse_page(page)
        return page
    
    def _parse_page(self, page: ScrapedPage):
        """Parse HTML content and extract structured data."""
        soup = BeautifulSoup(page.html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        page.title = title_tag.get_text(strip=True) if title_tag else "Untitled Page"
        
        # Extract clean text content
        page.text_content = self._extract_text(soup)
        page.word_count = len(page.text_content.split())
        
        # Extract tables
        page.tables = self._extract_tables(soup)
        page.table_count = len(page.tables)
        
        # Extract links
        page.links = self._extract_links(soup, page.url)
        
        # Extract metadata
        page.metadata.update(self._extract_metadata(soup))
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        
        # Get text with newlines for structure
        lines = []
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'td', 'th', 'span', 'div']):
            text = element.get_text(strip=True)
            if text and len(text) > 1:
                lines.append(text)
        
        # Deduplicate while preserving order
        seen = set()
        unique_lines = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return '\n'.join(unique_lines)
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[pd.DataFrame]:
        """Extract all tables from HTML as DataFrames."""
        tables = []
        
        for table_tag in soup.find_all('table'):
            try:
                # Extract headers
                headers = []
                thead = table_tag.find('thead')
                if thead:
                    header_row = thead.find('tr')
                    if header_row:
                        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
                
                # If no thead, try first row
                if not headers:
                    first_row = table_tag.find('tr')
                    if first_row and first_row.find('th'):
                        headers = [th.get_text(strip=True) for th in first_row.find_all('th')]
                
                # Extract body rows
                rows = []
                tbody = table_tag.find('tbody') or table_tag
                for tr in tbody.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td'])]
                    if cells and len(cells) > 0:
                        rows.append(cells)
                
                if rows:
                    if headers and len(headers) == len(rows[0]):
                        df = pd.DataFrame(rows, columns=headers)
                    else:
                        df = pd.DataFrame(rows)
                    
                    # Try to infer table subject from preceding heading
                    prev_heading = self._find_preceding_heading(table_tag)
                    if prev_heading:
                        df.attrs['title'] = prev_heading
                    
                    tables.append(df)
                    
            except Exception:
                continue
        
        return tables
    
    def _find_preceding_heading(self, tag: Tag) -> Optional[str]:
        """Find the heading that precedes a table element."""
        for sibling in tag.previous_siblings:
            if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                return sibling.get_text(strip=True)
        
        parent = tag.parent
        if parent:
            for sibling in parent.previous_siblings:
                if isinstance(sibling, Tag) and sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5']:
                    return sibling.get_text(strip=True)
        
        return None
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract relevant links from the page."""
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            text = a_tag.get_text(strip=True)
            
            # Resolve relative URLs
            if not href.startswith(('http://', 'https://', 'mailto:')):
                href = urljoin(base_url, href)
            
            if text and href.startswith('http'):
                links.append({
                    'text': text[:128],
                    'url': href
                })
        
        return links[:50]  # Limit to 50 links
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract page metadata from meta tags."""
        metadata = {}
        
        for meta in soup.find_all('meta'):
            name = meta.get('name', '') or meta.get('property', '')
            content = meta.get('content', '')
            if name and content:
                metadata[name] = content[:256]
        
        return metadata
