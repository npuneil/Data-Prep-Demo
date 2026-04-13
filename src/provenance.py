"""
Data Provenance Module - Tracks data lineage from source to output.

Maintains a complete audit trail of data transformations for
regulatory compliance and data governance requirements.
"""

import hashlib
import json
import platform
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone


@dataclass
class ProvenanceStep:
    """A single step in the data provenance chain."""
    step_number: int
    operation: str
    description: str
    timestamp: str
    input_hash: str = ""
    output_hash: str = ""
    records_in: int = 0
    records_out: int = 0
    duration_ms: float = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a data preparation job."""
    job_id: str
    created_at: str
    source_type: str                   # "web_page" or "pdf"
    source_url: str = ""
    source_filename: str = ""
    source_hash: str = ""
    target_schema: str = ""
    target_version: str = ""
    processing_device: str = ""
    processing_engine: str = ""
    steps: List[ProvenanceStep] = field(default_factory=list)
    output_format: str = ""
    output_hash: str = ""
    total_duration_ms: float = 0
    total_entities_extracted: int = 0
    total_records_produced: int = 0
    quality_score: float = 0
    environment: Dict[str, str] = field(default_factory=dict)


class ProvenanceTracker:
    """Tracks data provenance throughout the processing pipeline."""
    
    def __init__(self):
        self.current_record: Optional[ProvenanceRecord] = None
        self._step_counter = 0
    
    def start_job(self, source_type: str, source_url: str = "", 
                  source_filename: str = "", source_hash: str = "") -> ProvenanceRecord:
        """Initialize a new provenance tracking job."""
        job_id = self._generate_job_id(source_url or source_filename)
        
        self.current_record = ProvenanceRecord(
            job_id=job_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_type=source_type,
            source_url=source_url,
            source_filename=source_filename,
            source_hash=source_hash,
            environment={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or "unknown",
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }
        )
        self._step_counter = 0
        
        return self.current_record
    
    def add_step(self, operation: str, description: str,
                 input_hash: str = "", output_hash: str = "",
                 records_in: int = 0, records_out: int = 0,
                 duration_ms: float = 0, parameters: Dict[str, Any] = None,
                 warnings: List[str] = None) -> ProvenanceStep:
        """Add a processing step to the provenance chain."""
        self._step_counter += 1
        
        step = ProvenanceStep(
            step_number=self._step_counter,
            operation=operation,
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash=input_hash,
            output_hash=output_hash,
            records_in=records_in,
            records_out=records_out,
            duration_ms=duration_ms,
            parameters=parameters or {},
            warnings=warnings or []
        )
        
        if self.current_record:
            self.current_record.steps.append(step)
        
        return step
    
    def finalize(self, output_hash: str = "", total_records: int = 0,
                 quality_score: float = 0, output_format: str = "json") -> ProvenanceRecord:
        """Finalize the provenance record with output information."""
        if self.current_record:
            self.current_record.output_hash = output_hash
            self.current_record.total_records_produced = total_records
            self.current_record.quality_score = quality_score
            self.current_record.output_format = output_format
            self.current_record.total_duration_ms = sum(
                s.duration_ms for s in self.current_record.steps
            )
            self.current_record.total_entities_extracted = sum(
                s.records_out for s in self.current_record.steps
                if s.operation == "entity_extraction"
            )
        
        return self.current_record
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the current provenance record to a dictionary."""
        if self.current_record:
            return asdict(self.current_record)
        return {}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the current provenance record to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def _generate_job_id(self, source: str) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        hash_part = hashlib.md5(
            f"{source}_{timestamp}".encode()
        ).hexdigest()[:8]
        return f"MSDP_{timestamp}_{hash_part}"
