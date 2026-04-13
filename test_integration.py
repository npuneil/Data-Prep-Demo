"""Quick integration test for all modules."""
import sys
sys.path.insert(0, '.')

from src.npu_engine import NPUEngine
from src.scraper import WebScraper
from src.pdf_parser import PDFParser
from src.entity_extractor import EntityExtractor
from src.normalizer import DataNormalizer
from src.schema_mapper import SchemaMapper
from src.exporter import DataExporter
from src.provenance import ProvenanceTracker

# Quick integration test
engine = NPUEngine()
hw = engine.get_status_display()
print("Device:", hw["device"])
print("NPU:", hw["npu_available"])
print("Accelerator:", hw["accelerator"])

scraper = WebScraper()
page = scraper.fetch_sample()
print("Sample page:", page.title, "- words:", page.word_count, "tables:", page.table_count)

extractor = EntityExtractor()
result = extractor.extract(page.text_content, page.tables)
print("Extracted:", result.total_entities, "entities of", len(result.entity_counts), "types")
print("Entity types:", dict(result.entity_counts))

normalizer = DataNormalizer()
norm = normalizer.normalize(result.entities)
print("Normalized:", len(norm.normalized_records), "records, quality:", round(norm.overall_quality, 1))

mapper = SchemaMapper()
mapping = mapper.map_records(norm.normalized_records)
print("Mapped:", round(mapping.schema_coverage, 1), "% schema coverage")

exporter = DataExporter()
json_str = exporter.get_json_string(mapping.mapped_records, {}, mapper.get_schema_summary())
print("JSON export length:", len(json_str), "chars")

print()
print("ALL MODULES WORKING!")
