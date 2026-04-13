"""
PDF Parser Module - Extracts text, tables, and structure from PDF documents.

Uses PyMuPDF (fitz) for fast, local PDF processing. Handles both
text-based and scanned PDFs with layout-aware extraction.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class ParsedPDF:
    """Represents a parsed PDF document."""
    filename: str
    filepath: str = ""
    title: str = ""
    author: str = ""
    text_content: str = ""
    pages: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    page_count: int = 0
    word_count: int = 0
    table_count: int = 0
    content_hash: str = ""
    parse_timestamp: str = ""
    parse_duration_ms: float = 0
    has_images: bool = False
    file_size_bytes: int = 0


class PDFParser:
    """Parses PDF documents for financial data extraction."""
    
    def __init__(self):
        self._fitz_available = False
        try:
            import fitz
            self._fitz_available = True
        except ImportError:
            pass
    
    @property
    def is_available(self) -> bool:
        return self._fitz_available
    
    def parse_file(self, filepath: str) -> ParsedPDF:
        """Parse a PDF file from disk."""
        path = Path(filepath)
        
        result = ParsedPDF(
            filename=path.name,
            filepath=str(path.absolute()),
            parse_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        if not self._fitz_available:
            result.metadata["error"] = "PyMuPDF not installed"
            return result
        
        if not path.exists():
            result.metadata["error"] = f"File not found: {filepath}"
            return result
        
        result.file_size_bytes = path.stat().st_size
        
        with open(filepath, 'rb') as f:
            content = f.read()
            result.content_hash = hashlib.sha256(content).hexdigest()
        
        return self._parse_content(content, result)
    
    def parse_bytes(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> ParsedPDF:
        """Parse PDF from bytes (e.g., file upload)."""
        result = ParsedPDF(
            filename=filename,
            parse_timestamp=datetime.now(timezone.utc).isoformat(),
            file_size_bytes=len(pdf_bytes),
            content_hash=hashlib.sha256(pdf_bytes).hexdigest()
        )
        
        if not self._fitz_available:
            result.metadata["error"] = "PyMuPDF not installed"
            return result
        
        return self._parse_content(pdf_bytes, result)
    
    def _parse_content(self, content: bytes, result: ParsedPDF) -> ParsedPDF:
        """Parse PDF content using PyMuPDF."""
        import fitz
        
        start_time = time.perf_counter()
        
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            
            # Document metadata
            result.page_count = len(doc)
            meta = doc.metadata or {}
            result.title = meta.get('title', '') or result.filename
            result.author = meta.get('author', '')
            result.metadata = {
                'title': meta.get('title', ''),
                'author': meta.get('author', ''),
                'subject': meta.get('subject', ''),
                'creator': meta.get('creator', ''),
                'producer': meta.get('producer', ''),
                'creation_date': meta.get('creationDate', ''),
                'modification_date': meta.get('modDate', ''),
            }
            
            # Process each page
            all_text = []
            all_tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = {
                    'page_number': page_num + 1,
                    'width': page.rect.width,
                    'height': page.rect.height,
                    'text': '',
                    'blocks': [],
                    'has_images': False
                }
                
                # Extract text with layout
                text = page.get_text("text")
                page_data['text'] = text
                all_text.append(text)
                
                # Extract text blocks for structure
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block.get("type") == 0:  # Text block
                        block_text = ""
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                block_text += span.get("text", "")
                            block_text += "\n"
                        page_data['blocks'].append({
                            'type': 'text',
                            'text': block_text.strip(),
                            'bbox': block.get('bbox', [])
                        })
                    elif block.get("type") == 1:  # Image block
                        page_data['has_images'] = True
                        result.has_images = True
                
                # Attempt table extraction from page
                page_tables = self._extract_tables_from_page(page, text)
                for table in page_tables:
                    table.attrs['page'] = page_num + 1
                all_tables.extend(page_tables)
                
                result.pages.append(page_data)
            
            doc.close()
            
            result.text_content = '\n\n'.join(all_text)
            result.word_count = len(result.text_content.split())
            result.tables = all_tables
            result.table_count = len(all_tables)
            
        except Exception as e:
            result.metadata["error"] = str(e)
        
        result.parse_duration_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _extract_tables_from_page(self, page, text: str) -> List[pd.DataFrame]:
        """Extract table-like structures from a PDF page."""
        tables = []
        
        # Simple heuristic: look for lines with consistent column alignment
        lines = text.strip().split('\n')
        
        # Look for patterns that suggest tabular data
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_table and len(table_lines) > 1:
                    df = self._lines_to_dataframe(table_lines)
                    if df is not None:
                        tables.append(df)
                    table_lines = []
                    in_table = False
                continue
            
            # Heuristic: line has multiple number-like tokens = likely table row
            tokens = stripped.split()
            num_count = sum(1 for t in tokens if self._is_numeric_token(t))
            
            if num_count >= 2 or (in_table and num_count >= 1):
                in_table = True
                table_lines.append(stripped)
            elif in_table and len(tokens) >= 2:
                table_lines.append(stripped)
            else:
                if in_table and len(table_lines) > 1:
                    df = self._lines_to_dataframe(table_lines)
                    if df is not None:
                        tables.append(df)
                table_lines = []
                in_table = False
        
        # Handle remaining lines
        if in_table and len(table_lines) > 1:
            df = self._lines_to_dataframe(table_lines)
            if df is not None:
                tables.append(df)
        
        return tables
    
    def _lines_to_dataframe(self, lines: List[str]) -> Optional[pd.DataFrame]:
        """Convert aligned text lines to a DataFrame."""
        if len(lines) < 2:
            return None
        
        try:
            # Split each line by 2+ spaces (column separator heuristic)
            import re
            rows = []
            for line in lines:
                cells = re.split(r'\s{2,}', line.strip())
                if len(cells) >= 2:
                    rows.append(cells)
            
            if len(rows) < 2:
                return None
            
            # Normalize column count
            max_cols = max(len(r) for r in rows)
            for i, row in enumerate(rows):
                while len(row) < max_cols:
                    row.append('')
                rows[i] = row[:max_cols]
            
            # Use first row as header if it looks like one
            first_row = rows[0]
            is_header = sum(1 for c in first_row if not self._is_numeric_token(c)) > len(first_row) // 2
            
            if is_header:
                return pd.DataFrame(rows[1:], columns=first_row)
            else:
                return pd.DataFrame(rows)
                
        except Exception:
            return None
    
    def _is_numeric_token(self, token: str) -> bool:
        """Check if a token looks like a number."""
        cleaned = token.replace(',', '').replace('%', '').replace('$', '').replace('(', '').replace(')', '').replace('-', '').replace('+', '')
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
