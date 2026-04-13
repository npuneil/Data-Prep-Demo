"""
Entity Extractor Module - Extracts and classifies financial entities from text.

Uses a combination of rule-based patterns (always available, fast) and
optional ONNX-based ML extraction (NPU-accelerated when available).
Specialized for financial/investment data.
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime


@dataclass
class ExtractedEntity:
    """A single extracted entity with metadata."""
    entity_type: str          # e.g., "ticker", "isin", "fund_name", "nav", etc.
    value: str                # The raw extracted value
    normalized_value: Any     # Cleaned/normalized value
    confidence: float         # 0.0 - 1.0
    source_text: str          # The original text context
    position: Tuple[int, int] = (0, 0)  # Character position in source
    extraction_method: str = "rule_based"  # "rule_based" or "ml_model"
    category: str = ""        # Schema category
    mapped_field: str = ""    # Target schema field name


@dataclass
class ExtractionResult:
    """Result of entity extraction from a document."""
    entities: List[ExtractedEntity] = field(default_factory=list)
    entity_counts: Dict[str, int] = field(default_factory=dict)
    extraction_time_ms: float = 0
    method: str = "rule_based"
    text_length: int = 0
    total_entities: int = 0


# Common fund categories
FUND_CATEGORIES = [
    "Large Blend", "Large Growth", "Large Value",
    "Mid-Cap Blend", "Mid-Cap Growth", "Mid-Cap Value",
    "Small Blend", "Small Growth", "Small Value",
    "Foreign Large Blend", "Foreign Large Growth", "Foreign Large Value",
    "Diversified Emerging Mkts", "China Region", "India Equity",
    "Intermediate Core Bond", "Intermediate Core-Plus Bond",
    "Short-Term Bond", "Long-Term Bond", "High Yield Bond",
    "Inflation-Protected Bond", "Bank Loan",
    "Allocation--30% to 50% Equity", "Allocation--50% to 70% Equity",
    "Allocation--70% to 85% Equity", "Allocation--85%+ Equity",
    "Target-Date 2025", "Target-Date 2030", "Target-Date 2035",
    "Target-Date 2040", "Target-Date 2045", "Target-Date 2050",
    "Technology", "Health", "Financial", "Real Estate",
    "Natural Resources", "Utilities", "Communications",
    "Commodities Broad Basket", "Equity Precious Metals",
    "Money Market - Taxable", "Money Market - Tax-Free",
    "Multialternative", "Market Neutral", "Long-Short Equity",
]

# Known fund families
FUND_FAMILIES = [
    "Vanguard", "BlackRock", "Fidelity", "State Street", "Schwab",
    "T. Rowe Price", "Capital Group", "American Funds", "JP Morgan",
    "JPMorgan", "Morgan Stanley", "Goldman Sachs", "Invesco",
    "Franklin Templeton", "PIMCO", "Dimensional", "Dodge & Cox",
    "MFS", "Putnam", "Hartford", "John Hancock", "Principal",
    "Nuveen", "WisdomTree", "First Trust", "VanEck", "ProShares",
    "iShares", "SPDR", "Direxion", "Global X",
]

# Analyst ratings
ANALYST_RATINGS = ["Gold", "Silver", "Bronze", "Neutral", "Negative"]

# Asset class keywords
ASSET_CLASSES = {
    "Equity": ["equity", "stock", "shares", "growth", "value", "blend", "cap"],
    "Fixed Income": ["bond", "fixed income", "debt", "treasury", "corporate bond", "municipal"],
    "Allocation": ["allocation", "balanced", "target-date", "lifecycle"],
    "Alternative": ["alternative", "hedge", "long-short", "market neutral", "managed futures"],
    "Commodities": ["commodity", "commodities", "gold", "precious metals", "natural resources"],
    "Money Market": ["money market", "cash", "liquidity"],
}


class EntityExtractor:
    """Extracts financial entities from text using rule-based and ML approaches."""
    
    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for fast extraction."""
        self.patterns = {
            'isin': re.compile(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b'),
            'cusip': re.compile(r'\b([0-9]{3}[A-Z0-9]{5}[0-9])\b'),
            'ticker': re.compile(r'\b([A-Z]{1,5})\b(?=[\s,;)\]|$])'),
            'ticker_with_symbol': re.compile(r'(?:ticker|symbol)[:\s]+([A-Z]{1,5})\b', re.IGNORECASE),
            'ticker_parenthetical': re.compile(r'\(([A-Z]{2,5})\)'),
            'nav': re.compile(r'(?:NAV|Net\s+Asset\s+Value)[:\s]*\$?([\d,]+\.?\d*)', re.IGNORECASE),
            'expense_ratio': re.compile(r'(?:Expense\s+Ratio|ER)[:\s]*([\d.]+)\s*%', re.IGNORECASE),
            'yield_pct': re.compile(r'(?:SEC\s+)?(?:Yield|Distribution\s+Rate)[:\s]*([\d.]+)\s*%', re.IGNORECASE),
            'total_assets': re.compile(r'(?:Total\s+(?:Net\s+)?Assets|AUM)[:\s]*\$?([\d,.]+)\s*(Million|Billion|[MB])\b', re.IGNORECASE),
            'return_pct': re.compile(r'(-?[\d.]+)\s*%', re.IGNORECASE),
            'dollar_amount': re.compile(r'\$([\d,]+\.?\d*)\s*(million|billion|[mb])?\b', re.IGNORECASE),
            'date': re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b|\b(\w+\s+\d{1,2},?\s+\d{4})\b|\b(\d{4}-\d{2}-\d{2})\b'),
            'star_rating': re.compile(r'(?:Rating|Stars?)[:\s]*(?:([1-5])\s*(?:stars?|★)|([★]{1,5}))', re.IGNORECASE),
            'analyst_rating': re.compile(r'(?:Analyst\s+Rating|Medalist\s+Rating)[:\s]*(Gold|Silver|Bronze|Neutral|Negative)', re.IGNORECASE),
            'inception_date': re.compile(r'(?:Inception\s+Date|Established)[:\s]*(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})', re.IGNORECASE),
            'fund_name': re.compile(r'(?:^|\n)([A-Z][A-Za-z\s&.\']+(?:Fund|ETF|Trust|Portfolio|Index|Shares|Admiral|Investor|Institutional)(?:\s+[A-Z][a-z]+)*)', re.MULTILINE),
            'category': re.compile(r'(?:Category|Classification)[:\s]+([A-Za-z\s\-]+?)(?:\n|\||$)', re.IGNORECASE),
            'currency': re.compile(r'(?:Currency|Base Currency|Denomination)[:\s]*(USD|EUR|GBP|JPY|CHF|CAD|AUD|CNY|HKD|SGD)', re.IGNORECASE),
            'standard_deviation': re.compile(r'(?:Standard\s+Deviation|Std\.?\s*Dev\.?)[:\s]*([\d.]+)', re.IGNORECASE),
            'sharpe_ratio': re.compile(r'Sharpe\s+Ratio[:\s]*(-?[\d.]+)', re.IGNORECASE),
            'beta': re.compile(r'\bBeta[:\s]*(-?[\d.]+)', re.IGNORECASE),
            'alpha': re.compile(r'\bAlpha[:\s]*(-?[\d.]+)', re.IGNORECASE),
            'r_squared': re.compile(r'R-?Squared[:\s]*([\d.]+)', re.IGNORECASE),
        }
        
        # Performance return patterns with period context
        self.return_patterns = {
            '1M': re.compile(r'1[\s-]*(?:Month|Mo\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            '3M': re.compile(r'3[\s-]*(?:Month|Mo\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            'YTD': re.compile(r'(?:YTD|Year[\s-]*to[\s-]*Date)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            '1Y': re.compile(r'1[\s-]*(?:Year|Yr\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            '3Y': re.compile(r'3[\s-]*(?:Year|Yr\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            '5Y': re.compile(r'5[\s-]*(?:Year|Yr\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
            '10Y': re.compile(r'10[\s-]*(?:Year|Yr\.?)[:\s]*(-?[\d.]+)\s*%', re.IGNORECASE),
        }
    
    def extract(self, text: str, tables: list = None) -> ExtractionResult:
        """Extract all financial entities from text and tables."""
        start_time = time.perf_counter()
        result = ExtractionResult(text_length=len(text), method="rule_based")
        
        # Extract from free text
        text_entities = self._extract_from_text(text)
        result.entities.extend(text_entities)
        
        # Extract from tables
        if tables:
            table_entities = self._extract_from_tables(tables)
            result.entities.extend(table_entities)
        
        # Deduplicate entities
        result.entities = self._deduplicate_entities(result.entities)
        
        # Filter by confidence threshold
        result.entities = [e for e in result.entities if e.confidence >= self.confidence_threshold]
        
        # Count by type
        result.entity_counts = {}
        for entity in result.entities:
            result.entity_counts[entity.entity_type] = result.entity_counts.get(entity.entity_type, 0) + 1
        
        result.total_entities = len(result.entities)
        result.extraction_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    def _extract_from_text(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from free-form text."""
        entities = []
        
        # ISINs
        for match in self.patterns['isin'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="isin",
                value=match.group(1),
                normalized_value=match.group(1),
                confidence=0.95,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="identifiers",
                mapped_field="ISIN"
            ))
        
        # CUSIPs
        for match in self.patterns['cusip'].finditer(text):
            # Avoid matching ISINs that were already found
            isin_positions = {e.position for e in entities if e.entity_type == "isin"}
            if not any(match.start() >= s and match.end() <= e for s, e in isin_positions):
                entities.append(ExtractedEntity(
                    entity_type="cusip",
                    value=match.group(1),
                    normalized_value=match.group(1),
                    confidence=0.85,
                    source_text=text[max(0, match.start()-20):match.end()+20],
                    position=(match.start(), match.end()),
                    category="identifiers",
                    mapped_field="CUSIP"
                ))
        
        # Tickers (parenthetical - highest confidence)
        for match in self.patterns['ticker_parenthetical'].finditer(text):
            ticker = match.group(1)
            if len(ticker) >= 2 and not self._is_common_word(ticker):
                entities.append(ExtractedEntity(
                    entity_type="ticker",
                    value=ticker,
                    normalized_value=ticker,
                    confidence=0.90,
                    source_text=text[max(0, match.start()-30):match.end()+10],
                    position=(match.start(), match.end()),
                    category="identifiers",
                    mapped_field="Ticker"
                ))
        
        # Tickers (with keyword prefix)
        for match in self.patterns['ticker_with_symbol'].finditer(text):
            ticker = match.group(1)
            entities.append(ExtractedEntity(
                entity_type="ticker",
                value=ticker,
                normalized_value=ticker,
                confidence=0.92,
                source_text=text[max(0, match.start()-10):match.end()+10],
                position=(match.start(), match.end()),
                category="identifiers",
                mapped_field="Ticker"
            ))
        
        # Fund names
        for match in self.patterns['fund_name'].finditer(text):
            name = match.group(1).strip()
            if len(name) > 10:
                entities.append(ExtractedEntity(
                    entity_type="fund_name",
                    value=name,
                    normalized_value=name.strip(),
                    confidence=0.85,
                    source_text=text[max(0, match.start()-10):match.end()+10],
                    position=(match.start(), match.end()),
                    category="descriptive",
                    mapped_field="LegalName"
                ))
        
        # NAV
        for match in self.patterns['nav'].finditer(text):
            value = match.group(1).replace(',', '')
            entities.append(ExtractedEntity(
                entity_type="nav",
                value=f"${match.group(1)}",
                normalized_value=float(value),
                confidence=0.92,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="pricing",
                mapped_field="NAV"
            ))
        
        # Expense Ratio
        for match in self.patterns['expense_ratio'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="expense_ratio",
                value=f"{match.group(1)}%",
                normalized_value=float(match.group(1)),
                confidence=0.93,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="fees",
                mapped_field="ExpenseRatio"
            ))
        
        # Yield
        for match in self.patterns['yield_pct'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="yield",
                value=f"{match.group(1)}%",
                normalized_value=float(match.group(1)),
                confidence=0.90,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="income",
                mapped_field="Yield"
            ))
        
        # Total Net Assets
        for match in self.patterns['total_assets'].finditer(text):
            value = float(match.group(1).replace(',', ''))
            unit = match.group(2).upper()[0] if match.group(2) else 'M'
            if unit == 'B':
                value *= 1000  # Convert to millions
            entities.append(ExtractedEntity(
                entity_type="total_assets",
                value=f"${match.group(1)} {match.group(2) or 'M'}",
                normalized_value=value,
                confidence=0.88,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="size",
                mapped_field="TotalNetAssets"
            ))
        
        # Star Rating
        for match in self.patterns['star_rating'].finditer(text):
            rating = match.group(1) or str(len(match.group(2))) if match.group(2) else None
            if rating:
                entities.append(ExtractedEntity(
                    entity_type="star_rating",
                    value=f"{rating} stars",
                    normalized_value=int(rating),
                    confidence=0.95,
                    source_text=text[max(0, match.start()-20):match.end()+20],
                    position=(match.start(), match.end()),
                    category="ratings",
                    mapped_field="StarRating"
                ))
        
        # Analyst Rating
        for match in self.patterns['analyst_rating'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="analyst_rating",
                value=match.group(1),
                normalized_value=match.group(1),
                confidence=0.94,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="ratings",
                mapped_field="AnalystRating"
            ))
        
        # Inception Date
        for match in self.patterns['inception_date'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="inception_date",
                value=match.group(1),
                normalized_value=match.group(1),
                confidence=0.90,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="descriptive",
                mapped_field="InceptionDate"
            ))
        
        # Currency
        for match in self.patterns['currency'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="currency",
                value=match.group(1),
                normalized_value=match.group(1).upper(),
                confidence=0.92,
                source_text=text[max(0, match.start()-20):match.end()+20],
                position=(match.start(), match.end()),
                category="descriptive",
                mapped_field="BaseCurrency"
            ))
        
        # Risk Metrics
        for match in self.patterns['standard_deviation'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="standard_deviation",
                value=match.group(1),
                normalized_value=float(match.group(1)),
                confidence=0.88,
                source_text=text[max(0, match.start()-30):match.end()+10],
                position=(match.start(), match.end()),
                category="risk",
                mapped_field="StandardDeviation_3Y"
            ))
        
        for match in self.patterns['sharpe_ratio'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="sharpe_ratio",
                value=match.group(1),
                normalized_value=float(match.group(1)),
                confidence=0.88,
                source_text=text[max(0, match.start()-20):match.end()+10],
                position=(match.start(), match.end()),
                category="risk",
                mapped_field="SharpeRatio_3Y"
            ))
        
        for match in self.patterns['beta'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="beta",
                value=match.group(1),
                normalized_value=float(match.group(1)),
                confidence=0.87,
                source_text=text[max(0, match.start()-10):match.end()+10],
                position=(match.start(), match.end()),
                category="risk",
                mapped_field="Beta_3Y"
            ))
        
        for match in self.patterns['alpha'].finditer(text):
            entities.append(ExtractedEntity(
                entity_type="alpha",
                value=match.group(1),
                normalized_value=float(match.group(1)),
                confidence=0.87,
                source_text=text[max(0, match.start()-10):match.end()+10],
                position=(match.start(), match.end()),
                category="risk",
                mapped_field="Alpha_3Y"
            ))
        
        # Category
        for match in self.patterns['category'].finditer(text):
            cat_value = match.group(1).strip()
            # Verify against known categories
            matched_cat = None
            for known_cat in FUND_CATEGORIES:
                if known_cat.lower() in cat_value.lower() or cat_value.lower() in known_cat.lower():
                    matched_cat = known_cat
                    break
            if matched_cat:
                entities.append(ExtractedEntity(
                    entity_type="category",
                    value=cat_value,
                    normalized_value=matched_cat,
                    confidence=0.90,
                    source_text=text[max(0, match.start()-20):match.end()+20],
                    position=(match.start(), match.end()),
                    category="classification",
                    mapped_field="CategoryName"
                ))
        
        # Also check for categories mentioned in-line
        text_lower = text.lower()
        for known_cat in FUND_CATEGORIES:
            if known_cat.lower() in text_lower:
                idx = text_lower.index(known_cat.lower())
                entities.append(ExtractedEntity(
                    entity_type="category",
                    value=known_cat,
                    normalized_value=known_cat,
                    confidence=0.85,
                    source_text=text[max(0, idx-20):idx+len(known_cat)+20],
                    position=(idx, idx+len(known_cat)),
                    category="classification",
                    mapped_field="CategoryName"
                ))
        
        # Fund Family detection
        for family in FUND_FAMILIES:
            if family.lower() in text_lower:
                idx = text_lower.index(family.lower())
                entities.append(ExtractedEntity(
                    entity_type="fund_family",
                    value=family,
                    normalized_value=family,
                    confidence=0.92,
                    source_text=text[max(0, idx-10):idx+len(family)+10],
                    position=(idx, idx+len(family)),
                    category="descriptive",
                    mapped_field="FundFamily"
                ))
        
        # Asset class detection
        for asset_class, keywords in ASSET_CLASSES.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    idx = text_lower.index(keyword.lower())
                    entities.append(ExtractedEntity(
                        entity_type="asset_class",
                        value=asset_class,
                        normalized_value=asset_class,
                        confidence=0.75,
                        source_text=text[max(0, idx-20):idx+len(keyword)+20],
                        position=(idx, idx+len(keyword)),
                        category="classification",
                        mapped_field="GlobalAssetClassId"
                    ))
                    break  # Only need one match per asset class
        
        return entities
    
    def _extract_from_tables(self, tables: list) -> List[ExtractedEntity]:
        """Extract entities from structured table data."""
        entities = []
        
        for table_idx, table in enumerate(tables):
            if not hasattr(table, 'iterrows'):
                continue
            
            table_title = getattr(table, 'attrs', {}).get('title', f'Table {table_idx + 1}') if hasattr(table, 'attrs') else f'Table {table_idx + 1}'
            
            # Check if this is a performance table
            if any(kw in str(table_title).lower() for kw in ['performance', 'return']):
                entities.extend(self._extract_performance_from_table(table, table_title))
            
            # Check if this is a holdings table
            elif any(kw in str(table_title).lower() for kw in ['holding', 'portfolio', 'top']):
                entities.extend(self._extract_holdings_from_table(table, table_title))
            
            # Check if this is a risk metrics table
            elif any(kw in str(table_title).lower() for kw in ['risk', 'metric', 'statistic']):
                entities.extend(self._extract_risk_from_table(table, table_title))
            
            # Check if this is a sector allocation table
            elif any(kw in str(table_title).lower() for kw in ['sector', 'allocation', 'weight']):
                entities.extend(self._extract_sectors_from_table(table, table_title))
            
            # Generic extraction from any table
            else:
                entities.extend(self._extract_generic_from_table(table, table_title))
        
        return entities
    
    def _extract_performance_from_table(self, table, title: str) -> List[ExtractedEntity]:
        """Extract performance returns from a performance table."""
        entities = []
        
        period_map = {
            '1 month': ('Return_1M', '1M'), '1 mo': ('Return_1M', '1M'),
            '3 month': ('Return_3M', '3M'), '3 mo': ('Return_3M', '3M'),
            'ytd': ('Return_YTD', 'YTD'), 'year to date': ('Return_YTD', 'YTD'),
            '1 year': ('Return_1Y', '1Y'), '1 yr': ('Return_1Y', '1Y'),
            '3 year': ('Return_3Y', '3Y'), '3 yr': ('Return_3Y', '3Y'),
            '5 year': ('Return_5Y', '5Y'), '5 yr': ('Return_5Y', '5Y'),
            '10 year': ('Return_10Y', '10Y'), '10 yr': ('Return_10Y', '10Y'),
        }
        
        for _, row in table.iterrows():
            row_values = [str(v).strip() for v in row.values]
            row_text = ' '.join(row_values).lower()
            
            for period_key, (field_name, period_label) in period_map.items():
                if period_key in row_text:
                    # Find the percentage value in this row
                    for val in row_values:
                        match = re.match(r'^(-?[\d.]+)\s*%?$', val.strip())
                        if match:
                            entities.append(ExtractedEntity(
                                entity_type=f"return_{period_label}",
                                value=f"{match.group(1)}%",
                                normalized_value=float(match.group(1)),
                                confidence=0.90,
                                source_text=f"{title}: {' | '.join(row_values)}",
                                extraction_method="table_extraction",
                                category="performance",
                                mapped_field=field_name
                            ))
                            break  # Take first numeric value
        
        return entities
    
    def _extract_holdings_from_table(self, table, title: str) -> List[ExtractedEntity]:
        """Extract top holdings data from a holdings table."""
        entities = []
        holdings = []
        
        for _, row in table.iterrows():
            row_values = [str(v).strip() for v in row.values]
            
            # Look for ticker patterns and weights
            ticker = None
            name = None
            weight = None
            sector = None
            
            for val in row_values:
                if re.match(r'^[A-Z]{1,5}$', val) and not self._is_common_word(val):
                    ticker = val
                elif re.match(r'^[\d.]+$', val) and float(val) < 100:
                    if weight is None:
                        weight = float(val)
                elif len(val) > 5 and any(c.islower() for c in val):
                    if name is None:
                        name = val
                    else:
                        sector = val
            
            if name and weight:
                holdings.append({
                    'SecurityName': name,
                    'Ticker': ticker or '',
                    'Weight': weight,
                    'Sector': sector or ''
                })
        
        if holdings:
            entities.append(ExtractedEntity(
                entity_type="top_holdings",
                value=f"{len(holdings)} holdings",
                normalized_value=holdings,
                confidence=0.88,
                source_text=f"{title}: {len(holdings)} holdings extracted",
                extraction_method="table_extraction",
                category="holdings",
                mapped_field="TopHoldings"
            ))
        
        return entities
    
    def _extract_risk_from_table(self, table, title: str) -> List[ExtractedEntity]:
        """Extract risk metrics from a risk table."""
        entities = []
        
        metric_map = {
            'standard deviation': ('standard_deviation', 'StandardDeviation_3Y', 'risk'),
            'sharpe ratio': ('sharpe_ratio', 'SharpeRatio_3Y', 'risk'),
            'beta': ('beta', 'Beta_3Y', 'risk'),
            'alpha': ('alpha', 'Alpha_3Y', 'risk'),
            'r-squared': ('r_squared', 'RSquared_3Y', 'risk'),
        }
        
        for _, row in table.iterrows():
            row_values = [str(v).strip() for v in row.values]
            row_text = ' '.join(row_values).lower()
            
            for metric_key, (entity_type, field_name, category) in metric_map.items():
                if metric_key in row_text:
                    for val in row_values:
                        match = re.match(r'^(-?[\d.]+)$', val.strip())
                        if match:
                            entities.append(ExtractedEntity(
                                entity_type=entity_type,
                                value=match.group(1),
                                normalized_value=float(match.group(1)),
                                confidence=0.90,
                                source_text=f"{title}: {' | '.join(row_values)}",
                                extraction_method="table_extraction",
                                category=category,
                                mapped_field=field_name
                            ))
                            break
        
        return entities
    
    def _extract_sectors_from_table(self, table, title: str) -> List[ExtractedEntity]:
        """Extract sector allocation from a sector table."""
        sectors = {}
        
        for _, row in table.iterrows():
            row_values = [str(v).strip() for v in row.values]
            
            sector_name = None
            weight = None
            
            for val in row_values:
                if re.match(r'^[\d.]+$', val):
                    if weight is None:
                        weight = float(val)
                elif len(val) > 2 and any(c.isalpha() for c in val):
                    if sector_name is None:
                        sector_name = val
            
            if sector_name and weight:
                sectors[sector_name] = weight
        
        if sectors:
            return [ExtractedEntity(
                entity_type="sector_allocation",
                value=f"{len(sectors)} sectors",
                normalized_value=sectors,
                confidence=0.87,
                source_text=f"{title}: {len(sectors)} sectors",
                extraction_method="table_extraction",
                category="allocation",
                mapped_field="SectorWeights"
            )]
        
        return []
    
    def _extract_generic_from_table(self, table, title: str) -> List[ExtractedEntity]:
        """Generic extraction from any table."""
        entities = []
        
        for _, row in table.iterrows():
            row_values = [str(v).strip() for v in row.values]
            row_text = ' '.join(row_values)
            
            # Extract any ISINs, tickers, or significant numbers
            for val in row_values:
                if re.match(r'^[A-Z]{2}[A-Z0-9]{9}[0-9]$', val):
                    entities.append(ExtractedEntity(
                        entity_type="isin",
                        value=val,
                        normalized_value=val,
                        confidence=0.90,
                        source_text=row_text,
                        extraction_method="table_extraction",
                        category="identifiers",
                        mapped_field="ISIN"
                    ))
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        seen = {}
        for entity in entities:
            key = (entity.entity_type, str(entity.normalized_value))
            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity
        return list(seen.values())
    
    def _is_common_word(self, word: str) -> bool:
        """Check if an uppercase word is a common English word (not a ticker)."""
        common = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
            'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS',
            'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY',
            'WHO', 'DID', 'GET', 'LET', 'SAY', 'SHE', 'TOO', 'USE',
            'INC', 'LTD', 'PLC', 'ETF', 'NAV', 'SEC', 'USD', 'EUR',
            'ANN', 'AVG', 'YTD', 'NET', 'PER', 'ADV',
            'FUND', 'BOND', 'DATE', 'RATE', 'YEAR', 'FROM', 'WITH',
            'THAT', 'THIS', 'HAVE', 'BEEN', 'WILL', 'MORE', 'WHEN',
            'SOME', 'THAN', 'THEM', 'THEN', 'WHAT', 'YOUR', 'EACH',
        }
        return word in common
