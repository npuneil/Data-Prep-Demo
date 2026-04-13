"""
Exporter Module - Generates clean, schema-aligned outputs with provenance.

Produces JSON and CSV exports formatted for data ingestion,
complete with data lineage and quality metadata.
"""

import csv
import hashlib
import json
import io
from dataclasses import asdict
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class DataExporter:
    """Exports processed data in formats ready for ingestion."""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(self, mapped_records: List[Dict[str, Any]], 
                    provenance: Dict[str, Any] = None,
                    schema_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export data as a JSON package."""
        
        # Clean records - remove internal metadata for export
        clean_records = []
        for record in mapped_records:
            clean = {k: v for k, v in record.items() if not k.startswith('_')}
            clean_records.append(clean)
        
        # Build the export package
        export_package = {
            "data_import": {
                "format_version": "1.0",
                "schema": schema_info.get("name", "Security Master") if schema_info else "Security Master",
                "schema_version": schema_info.get("version", "3.2.1") if schema_info else "3.2.1",
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "record_count": len(clean_records),
                "data": clean_records
            },
            "quality_metrics": {
                "overall_score": provenance.get("quality_score", 0) if provenance else 0,
                "records_processed": len(clean_records),
                "field_coverage": self._calculate_coverage(clean_records, schema_info)
            },
            "provenance": provenance or {}
        }
        
        # Calculate content hash
        content_str = json.dumps(export_package["data_import"]["data"], sort_keys=True)
        export_package["content_hash"] = hashlib.sha256(content_str.encode()).hexdigest()
        
        return export_package
    
    def export_csv(self, mapped_records: List[Dict[str, Any]]) -> str:
        """Export data as CSV string."""
        if not mapped_records:
            return ""
        
        # Clean records
        clean_records = []
        for record in mapped_records:
            clean = {}
            for k, v in record.items():
                if k.startswith('_'):
                    continue
                # Flatten complex types
                if isinstance(v, (list, dict)):
                    clean[k] = json.dumps(v)
                else:
                    clean[k] = v
            clean_records.append(clean)
        
        # Get all column names (union of all records)
        columns = []
        seen_cols = set()
        for record in clean_records:
            for key in record:
                if key not in seen_cols:
                    columns.append(key)
                    seen_cols.add(key)
        
        # Build CSV
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for record in clean_records:
            writer.writerow(record)
        
        return output.getvalue()
    
    def export_to_files(self, mapped_records: List[Dict[str, Any]],
                        provenance: Dict[str, Any] = None,
                        schema_info: Dict[str, Any] = None,
                        base_filename: str = None) -> Dict[str, str]:
        """Export both JSON and CSV files to the output directory."""
        if base_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"data_prep_export_{timestamp}"
        
        files = {}
        
        # JSON export
        json_package = self.export_json(mapped_records, provenance, schema_info)
        json_path = self.output_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_package, f, indent=2, default=str)
        files['json'] = str(json_path)
        
        # CSV export
        csv_content = self.export_csv(mapped_records)
        csv_path = self.output_dir / f"{base_filename}.csv"
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        files['csv'] = str(csv_path)
        
        # Provenance file
        if provenance:
            prov_path = self.output_dir / f"{base_filename}_provenance.json"
            with open(prov_path, 'w', encoding='utf-8') as f:
                json.dump(provenance, f, indent=2, default=str)
            files['provenance'] = str(prov_path)
        
        return files
    
    def get_json_string(self, mapped_records: List[Dict[str, Any]],
                        provenance: Dict[str, Any] = None,
                        schema_info: Dict[str, Any] = None) -> str:
        """Get JSON export as a string (for download buttons)."""
        package = self.export_json(mapped_records, provenance, schema_info)
        return json.dumps(package, indent=2, default=str)
    
    def get_csv_string(self, mapped_records: List[Dict[str, Any]]) -> str:
        """Get CSV export as a string (for download buttons)."""
        return self.export_csv(mapped_records)
    
    def _calculate_coverage(self, records: List[Dict[str, Any]], 
                           schema_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate field coverage statistics."""
        if not records:
            return {"total_fields": 0, "populated_fields": 0, "coverage_pct": 0}
        
        all_fields = set()
        populated_fields = set()
        
        for record in records:
            for key, value in record.items():
                all_fields.add(key)
                if value is not None and value != "":
                    populated_fields.add(key)
        
        total_schema_fields = schema_info.get("total_fields", len(all_fields)) if schema_info else len(all_fields)
        
        return {
            "total_schema_fields": total_schema_fields,
            "populated_fields": len(populated_fields),
            "coverage_pct": round(len(populated_fields) / total_schema_fields * 100, 1) if total_schema_fields > 0 else 0,
            "field_list": sorted(populated_fields)
        }
