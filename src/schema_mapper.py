"""
Schema Mapper Module - Maps extracted data to target schema.

Handles field mapping, type conversion, schema validation,
and generation of mapping reports for the data ingestion pipeline.
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path


@dataclass
class FieldMapping:
    """Mapping between an extracted field and a schema field."""
    source_field: str
    target_field: str
    target_display_name: str
    data_type: str
    value: Any
    is_valid: bool = True
    validation_message: str = ""
    confidence: float = 0.0
    category: str = ""


@dataclass
class SchemaMappingResult:
    """Result of mapping extracted data to the target schema."""
    mappings: List[FieldMapping] = field(default_factory=list)
    mapped_records: List[Dict[str, Any]] = field(default_factory=list)
    unmapped_fields: List[str] = field(default_factory=list)
    schema_coverage: float = 0.0
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)
    schema_name: str = ""
    schema_version: str = ""


class SchemaMapper:
    """Maps normalized data to the target schema."""
    
    def __init__(self, schema_path: str = None):
        self.schema = self._load_schema(schema_path)
        self.field_index = self._build_field_index()
    
    def _load_schema(self, schema_path: str = None) -> Dict[str, Any]:
        """Load the target schema definition."""
        if schema_path and Path(schema_path).exists():
            with open(schema_path, 'r') as f:
                return json.load(f)
        
        # Try default path
        default_path = Path(__file__).parent.parent / "config" / "schema.json"
        if default_path.exists():
            with open(default_path, 'r') as f:
                return json.load(f)
        
        # Embedded minimal schema
        return {
            "schema_name": "Security Master",
            "schema_version": "3.2.1",
            "fields": []
        }
    
    def _build_field_index(self) -> Dict[str, Dict[str, Any]]:
        """Build a lookup index of schema fields."""
        index = {}
        for field_def in self.schema.get("fields", []):
            index[field_def["field_name"]] = field_def
        return index
    
    def map_records(self, normalized_records: List[Dict[str, Any]]) -> SchemaMappingResult:
        """Map normalized records to the target schema."""
        result = SchemaMappingResult(
            schema_name=self.schema.get("schema_name", "Unknown"),
            schema_version=self.schema.get("schema_version", "0.0.0")
        )
        
        all_source_fields = set()
        
        for record in normalized_records:
            mapped_record, mappings, errors = self._map_single_record(record)
            result.mapped_records.append(mapped_record)
            result.mappings.extend(mappings)
            result.validation_errors.extend(errors)
            
            for key in record:
                if not key.startswith('_'):
                    all_source_fields.add(key)
        
        # Find unmapped fields
        mapped_targets = {m.target_field for m in result.mappings}
        for field_name in self.field_index:
            if field_name not in mapped_targets:
                result.unmapped_fields.append(field_name)
        
        # Calculate schema coverage
        total_schema_fields = len(self.field_index)
        if total_schema_fields > 0:
            result.schema_coverage = round(
                len(mapped_targets) / total_schema_fields * 100, 1
            )
        
        result.validation_passed = len(result.validation_errors) == 0
        
        return result
    
    def _map_single_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[FieldMapping], List[str]]:
        """Map a single normalized record to the target schema."""
        mapped = {}
        mappings = []
        errors = []
        
        for field_name, field_def in self.field_index.items():
            if field_name in record:
                value = record[field_name]
                
                # Type conversion and validation
                converted_value, is_valid, message = self._convert_and_validate(
                    value, field_def
                )
                
                if is_valid:
                    mapped[field_name] = converted_value
                else:
                    if field_def.get("required", False):
                        errors.append(f"Required field '{field_name}': {message}")
                    mapped[field_name] = converted_value  # Include even if invalid
                
                mappings.append(FieldMapping(
                    source_field=field_name,
                    target_field=field_name,
                    target_display_name=field_def.get("display_name", field_name),
                    data_type=field_def.get("data_type", "string"),
                    value=converted_value,
                    is_valid=is_valid,
                    validation_message=message,
                    confidence=record.get('_metadata', {}).get('avg_confidence', 0.8),
                    category=field_def.get("category", "")
                ))
            
            elif field_def.get("required", False):
                errors.append(f"Required field '{field_name}' is missing")
                mappings.append(FieldMapping(
                    source_field="",
                    target_field=field_name,
                    target_display_name=field_def.get("display_name", field_name),
                    data_type=field_def.get("data_type", "string"),
                    value=None,
                    is_valid=False,
                    validation_message="Required field missing",
                    category=field_def.get("category", "")
                ))
        
        # Include metadata
        if '_metadata' in record:
            mapped['_metadata'] = record['_metadata']
        
        return mapped, mappings, errors
    
    def _convert_and_validate(self, value: Any, field_def: Dict[str, Any]) -> Tuple[Any, bool, str]:
        """Convert a value to the target type and validate."""
        data_type = field_def.get("data_type", "string")
        
        if value is None:
            if field_def.get("required", False):
                return None, False, "Value is null for required field"
            return None, True, ""
        
        try:
            if data_type == "string":
                converted = str(value)
                max_length = field_def.get("max_length")
                if max_length and len(converted) > max_length:
                    converted = converted[:max_length]
                    return converted, True, f"Truncated to {max_length} chars"
                
                # Pattern validation
                pattern = field_def.get("pattern")
                if pattern:
                    import re
                    if not re.match(pattern, converted):
                        return converted, False, f"Does not match pattern {pattern}"
                
                # Allowed values
                allowed = field_def.get("allowed_values")
                if allowed and converted not in allowed:
                    return converted, False, f"Not in allowed values: {allowed}"
                
                return converted, True, ""
            
            elif data_type == "decimal":
                if isinstance(value, str):
                    value = float(value.replace(',', '').replace('$', '').replace('%', ''))
                converted = float(value)
                precision = field_def.get("precision", 4)
                converted = round(converted, precision)
                
                min_val = field_def.get("min_value")
                max_val = field_def.get("max_value")
                if min_val is not None and converted < min_val:
                    return converted, False, f"Below minimum {min_val}"
                if max_val is not None and converted > max_val:
                    return converted, False, f"Above maximum {max_val}"
                
                return converted, True, ""
            
            elif data_type == "integer":
                converted = int(float(value)) if isinstance(value, str) else int(value)
                
                min_val = field_def.get("min_value")
                max_val = field_def.get("max_value")
                if min_val is not None and converted < min_val:
                    return converted, False, f"Below minimum {min_val}"
                if max_val is not None and converted > max_val:
                    return converted, False, f"Above maximum {max_val}"
                
                return converted, True, ""
            
            elif data_type == "date":
                converted = str(value)
                # Validate date format
                expected_format = field_def.get("format", "YYYY-MM-DD")
                if expected_format == "YYYY-MM-DD":
                    import re
                    if not re.match(r'^\d{4}-\d{2}-\d{2}$', converted):
                        return converted, False, f"Expected format {expected_format}"
                
                return converted, True, ""
            
            elif data_type in ("array", "object"):
                return value, True, ""
            
            else:
                return value, True, ""
        
        except (ValueError, TypeError) as e:
            return value, False, f"Conversion error: {str(e)}"
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the target schema."""
        categories = {}
        for field_def in self.schema.get("fields", []):
            cat = field_def.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "field": field_def["field_name"],
                "display": field_def.get("display_name", field_def["field_name"]),
                "type": field_def.get("data_type", "string"),
                "required": field_def.get("required", False)
            })
        
        return {
            "name": self.schema.get("schema_name", "Unknown"),
            "version": self.schema.get("schema_version", "0.0.0"),
            "description": self.schema.get("description", ""),
            "total_fields": len(self.schema.get("fields", [])),
            "required_fields": len([f for f in self.schema.get("fields", []) if f.get("required")]),
            "categories": categories,
            "category_names": self.schema.get("categories", {})
        }
