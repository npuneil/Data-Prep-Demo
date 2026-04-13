"""
Normalizer Module - Standardizes and validates extracted financial data.

Handles date format normalization, currency standardization, percentage
formatting, data quality scoring, and validation against expected ranges.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, date


@dataclass
class NormalizationResult:
    """Result of normalizing extracted entities."""
    normalized_records: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    overall_quality: float = 0.0
    field_coverage: Dict[str, bool] = field(default_factory=dict)


# Expected ranges for financial metrics (for validation)
VALIDATION_RANGES = {
    'NAV': (0.01, 100000),
    'ExpenseRatio': (0.0, 10.0),
    'Yield': (-5.0, 30.0),
    'TotalNetAssets': (0.1, 5000000),  # In millions
    'Return_1M': (-50, 50),
    'Return_3M': (-60, 100),
    'Return_YTD': (-80, 200),
    'Return_1Y': (-80, 300),
    'Return_3Y': (-50, 100),
    'Return_5Y': (-30, 80),
    'Return_10Y': (-20, 50),
    'StandardDeviation_3Y': (0, 100),
    'SharpeRatio_3Y': (-5, 10),
    'Beta_3Y': (-3, 5),
    'Alpha_3Y': (-50, 50),
    'StarRating': (1, 5),
}


class DataNormalizer:
    """Normalizes extracted financial data for ingestion."""
    
    def __init__(self, date_format: str = "%Y-%m-%d", currency_default: str = "USD"):
        self.date_format = date_format
        self.currency_default = currency_default
    
    def normalize(self, entities: list) -> NormalizationResult:
        """Normalize a list of extracted entities into clean records."""
        result = NormalizationResult()
        
        # Group entities by security (fund)
        securities = self._group_by_security(entities)
        
        for sec_id, sec_entities in securities.items():
            record = self._build_record(sec_entities)
            
            # Validate the record
            warnings, errors = self._validate_record(record)
            result.warnings.extend(warnings)
            result.errors.extend(errors)
            
            # Calculate quality score for this record
            quality = self._score_record_quality(record)
            result.quality_scores[sec_id] = quality
            
            result.normalized_records.append(record)
        
        # Overall quality metrics
        if result.quality_scores:
            result.overall_quality = sum(result.quality_scores.values()) / len(result.quality_scores)
        
        result.completeness_score = self._calculate_completeness(result.normalized_records)
        result.accuracy_score = self._calculate_accuracy(result.normalized_records, result.errors)
        result.field_coverage = self._calculate_field_coverage(result.normalized_records)
        
        return result
    
    def _group_by_security(self, entities: list) -> Dict[str, list]:
        """Group entities by inferred security identity using ISIN boundaries."""
        # First, try to group by ISIN (strongest identifier)
        isin_entities = [e for e in entities if e.entity_type == "isin"]
        
        if len(isin_entities) >= 2:
            return self._group_by_identifier(entities, "isin")
        
        # Fallback: try grouping by ticker
        ticker_entities = [e for e in entities if e.entity_type == "ticker"]
        if len(ticker_entities) >= 2:
            return self._group_by_identifier(entities, "ticker")
        
        # Fallback: try grouping by fund_name
        name_entities = [e for e in entities if e.entity_type == "fund_name"]
        if len(name_entities) >= 2:
            return self._group_by_identifier(entities, "fund_name")
        
        # No grouping possible — single record
        return {"security_001": entities}
    
    def _group_by_identifier(self, entities: list, id_type: str) -> Dict[str, list]:
        """Group entities based on position relative to identifier entities."""
        # Get positions (indices) of the identifier entities
        id_indices = [i for i, e in enumerate(entities) if e.entity_type == id_type]
        
        if len(id_indices) <= 1:
            return {"security_001": entities}
        
        # Split at each identifier occurrence
        groups = {}
        for g_idx in range(len(id_indices)):
            start = id_indices[g_idx]
            end = id_indices[g_idx + 1] if g_idx + 1 < len(id_indices) else len(entities)
            group_entities = entities[start:end]
            
            # Prepend any entities before the first identifier to the first group
            if g_idx == 0 and id_indices[0] > 0:
                group_entities = entities[:id_indices[0]] + group_entities
            
            groups[f"security_{g_idx + 1:03d}"] = group_entities
        
        return groups
    
    def _build_record(self, entities: list) -> Dict[str, Any]:
        """Build a normalized record from a group of entities."""
        record = {
            '_metadata': {
                'entity_count': len(entities),
                'extraction_methods': list(set(e.extraction_method for e in entities)),
                'avg_confidence': sum(e.confidence for e in entities) / len(entities) if entities else 0
            }
        }
        
        for entity in entities:
            field_name = entity.mapped_field
            if not field_name:
                continue
            
            value = entity.normalized_value
            
            # Apply type-specific normalization
            if entity.entity_type in ('inception_date',) or 'Date' in field_name:
                value = self._normalize_date(value)
            elif field_name in ('NAV', 'ExpenseRatio', 'Yield', 'TotalNetAssets'):
                value = self._normalize_numeric(value, field_name)
            elif 'Return' in field_name:
                value = self._normalize_percentage(value)
            elif field_name in ('StandardDeviation_3Y', 'SharpeRatio_3Y', 'Beta_3Y', 'Alpha_3Y'):
                value = self._normalize_numeric(value, field_name)
            elif field_name == 'BaseCurrency':
                value = str(value).upper()[:3]
            elif field_name == 'StarRating':
                value = self._normalize_star_rating(value)
            
            # Only set if not already set, or if higher confidence
            if field_name not in record or entity.confidence > record.get(f'_conf_{field_name}', 0):
                record[field_name] = value
                record[f'_conf_{field_name}'] = entity.confidence
        
        # Set defaults
        if 'BaseCurrency' not in record:
            record['BaseCurrency'] = self.currency_default
        
        # Generate a SecId if we have enough identifiers
        if 'SecId' not in record:
            record['SecId'] = self._generate_sec_id(record)
        
        # Clean up confidence tracking keys
        conf_keys = [k for k in record if k.startswith('_conf_')]
        for k in conf_keys:
            del record[k]
        
        return record
    
    def _normalize_date(self, value: Any) -> Optional[str]:
        """Normalize various date formats to YYYY-MM-DD."""
        if isinstance(value, (date, datetime)):
            return value.strftime(self.date_format)
        
        if not isinstance(value, str):
            return None
        
        value = value.strip()
        
        # Try various date formats
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%B %d, %Y',
            '%B %d %Y',
            '%b %d, %Y',
            '%b %d %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%m/%d/%y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime(self.date_format)
            except ValueError:
                continue
        
        return value  # Return as-is if can't parse
    
    def _normalize_numeric(self, value: Any, field_name: str = "") -> Optional[float]:
        """Normalize numeric values."""
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        
        if isinstance(value, str):
            # Remove currency symbols, commas, spaces
            cleaned = value.replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            try:
                return round(float(cleaned), 4)
            except ValueError:
                return None
        
        return None
    
    def _normalize_percentage(self, value: Any) -> Optional[float]:
        """Normalize percentage values."""
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        
        if isinstance(value, str):
            cleaned = value.replace('%', '').replace(',', '').strip()
            try:
                return round(float(cleaned), 4)
            except ValueError:
                return None
        
        return None
    
    def _normalize_star_rating(self, value: Any) -> Optional[int]:
        """Normalize star ratings to 1-5 integer."""
        if isinstance(value, int):
            return max(1, min(5, value))
        if isinstance(value, float):
            return max(1, min(5, int(round(value))))
        if isinstance(value, str):
            # Count stars
            stars = value.count('★') or value.count('*')
            if stars:
                return max(1, min(5, stars))
            try:
                return max(1, min(5, int(float(value))))
            except ValueError:
                return None
        return None
    
    def _generate_sec_id(self, record: Dict[str, Any]) -> str:
        """Generate a security ID."""
        import hashlib
        
        # Use available identifiers to create a deterministic ID
        id_parts = []
        for field in ['ISIN', 'CUSIP', 'Ticker', 'LegalName']:
            if field in record:
                id_parts.append(str(record[field]))
        
        if id_parts:
            hash_input = '|'.join(id_parts)
            hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:8].upper()
            return f"F{hash_val}0"
        
        return "F00000000"
    
    def _validate_record(self, record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate a record against expected ranges and rules."""
        warnings = []
        errors = []
        
        for field_name, (min_val, max_val) in VALIDATION_RANGES.items():
            if field_name in record and record[field_name] is not None:
                value = record[field_name]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        warnings.append(
                            f"{field_name}: value {value} outside expected range [{min_val}, {max_val}]"
                        )
        
        # Required field checks
        if 'LegalName' not in record and 'Ticker' not in record:
            errors.append("Record has no fund name or ticker identifier")
        
        return warnings, errors
    
    def _score_record_quality(self, record: Dict[str, Any]) -> float:
        """Calculate a quality score (0-100) for a record."""
        score = 0
        max_score = 0
        
        # Identifiers (25 points)
        id_fields = ['ISIN', 'CUSIP', 'Ticker', 'LegalName', 'SecId']
        for f in id_fields:
            max_score += 5
            if f in record and record[f]:
                score += 5
        
        # Descriptive (15 points)
        desc_fields = ['FundFamily', 'CategoryName', 'InceptionDate', 'BaseCurrency', 'GlobalAssetClassId']
        for f in desc_fields:
            max_score += 3
            if f in record and record[f]:
                score += 3
        
        # Pricing (10 points)
        for f in ['NAV', 'TotalNetAssets']:
            max_score += 5
            if f in record and record[f]:
                score += 5
        
        # Fees (5 points)
        for f in ['ExpenseRatio']:
            max_score += 5
            if f in record and record[f]:
                score += 5
        
        # Performance (25 points)
        perf_fields = ['Return_1M', 'Return_3M', 'Return_YTD', 'Return_1Y', 'Return_3Y', 'Return_5Y', 'Return_10Y']
        for f in perf_fields:
            max_score += 3.57
            if f in record and record[f] is not None:
                score += 3.57
        
        # Risk (15 points)
        risk_fields = ['StandardDeviation_3Y', 'SharpeRatio_3Y', 'Beta_3Y', 'Alpha_3Y']
        for f in risk_fields:
            max_score += 3.75
            if f in record and record[f] is not None:
                score += 3.75
        
        # Ratings (5 points)
        for f in ['StarRating', 'AnalystRating']:
            max_score += 2.5
            if f in record and record[f]:
                score += 2.5
        
        return round((score / max_score * 100) if max_score > 0 else 0, 1)
    
    def _calculate_completeness(self, records: List[Dict[str, Any]]) -> float:
        """Calculate overall data completeness score."""
        if not records:
            return 0.0
        
        required_fields = ['SecId', 'LegalName', 'BaseCurrency']
        optional_fields = [
            'ISIN', 'CUSIP', 'Ticker', 'FundFamily', 'CategoryName',
            'InceptionDate', 'NAV', 'TotalNetAssets', 'ExpenseRatio',
            'Return_1Y', 'Return_3Y', 'Return_5Y',
            'StandardDeviation_3Y', 'SharpeRatio_3Y', 'StarRating'
        ]
        
        total_fields = len(required_fields) + len(optional_fields)
        filled = 0
        
        for record in records:
            for f in required_fields:
                if f in record and record[f]:
                    filled += 1
            for f in optional_fields:
                if f in record and record[f] is not None:
                    filled += 1
        
        return round(filled / (total_fields * len(records)) * 100, 1)
    
    def _calculate_accuracy(self, records: List[Dict[str, Any]], errors: List[str]) -> float:
        """Calculate accuracy score based on validation results."""
        if not records:
            return 0.0
        
        total_fields = sum(len([k for k in r.keys() if not k.startswith('_')]) for r in records)
        error_count = len(errors)
        
        if total_fields == 0:
            return 100.0
        
        return round(max(0, (1 - error_count / total_fields) * 100), 1)
    
    def _calculate_field_coverage(self, records: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Calculate which schema fields are covered."""
        all_fields = set()
        for record in records:
            for key in record:
                if not key.startswith('_'):
                    all_fields.add(key)
        
        schema_fields = [
            'SecId', 'ISIN', 'CUSIP', 'Ticker', 'LegalName', 'FundFamily',
            'CategoryName', 'GlobalAssetClassId', 'InceptionDate', 'BaseCurrency',
            'NAV', 'NAVDate', 'TotalNetAssets', 'ExpenseRatio', 'Yield',
            'Return_1M', 'Return_3M', 'Return_YTD', 'Return_1Y',
            'Return_3Y', 'Return_5Y', 'Return_10Y',
            'StandardDeviation_3Y', 'SharpeRatio_3Y', 'Beta_3Y', 'Alpha_3Y',
            'StarRating', 'AnalystRating', 'StyleBox',
            'TopHoldings', 'SectorWeights'
        ]
        
        return {f: f in all_fields for f in schema_fields}
