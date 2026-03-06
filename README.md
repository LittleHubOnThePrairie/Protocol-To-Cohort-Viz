# Protocol-To-Cohort-Viz (PTCV)

> ML-enabled clinical trial software for parsing, classifying, and transforming clinical trial
> protocols into CDISC-compliant SDTM datasets with interactive Schedule of Activities
> visualizations.

## Overview

PTCV ingests clinical trial protocols from public registries (EU-CTR / CTIS and ClinicalTrials.gov),
extracts structured content from PDF and CTR-XML documents, classifies sections against the
ICH E6(R3) Appendix B format, builds a CDISC USDM v4.0 Schedule of Activities model, and
generates SDTM trial design datasets (TS, TA, TE, TV, TI) with accompanying Define-XML v2.1.
All artifact writes are immutable (WORM) with a tamper-evident SHA-256 lineage chain, satisfying
21 CFR Part 11, ICH E6(R3), and ALCOA+ requirements.

The project ships a Streamlit application with two independent processing pipelines:

- **Process (LLM Retemplater)** — rule-based ICH section classification followed by optional
  LLM-powered retemplating against the full ICH E6(R3) Appendix B structure.
- **Query Pipeline** — query-driven extraction using a pre-defined schema of questions per
  section, with optional Claude Sonnet transformation for content reframing.

Both pipelines feed the SoA extraction and SDTM generation stages.

## Architecture

```
src/ptcv/
    protocol_search/    — Search and download protocols from EU-CTR and ClinicalTrials.gov
    extraction/         — Format detection and text/table extraction from PDF and CTR-XML
    ich_parser/         — ICH E6(R3) section classifier, LLM retemplater, query extractor,
                          and template assembler
    soa_extractor/      — Schedule of Activities parser, CDISC USDM v4.0 mapper, and
                          query pipeline bridge
    sdtm/               — SDTM domain generators, Define-XML, CT normalizer, and validator
    annotations/        — Span-level annotation service for section boundaries
    storage/            — StorageGateway abstraction (FilesystemAdapter / MinIO WORM)
    compliance/         — Audit trail, data integrity, and access control (21 CFR Part 11)
    pipeline/           — End-to-end PipelineOrchestrator wiring all stages
    ui/                 — Streamlit application with modular component library
```

### Pipeline stages

1. **protocol_search** — retrieves protocol PDFs or CTR-XML from registries, stores via
   `StorageGateway` with SHA-256 checksums.
2. **extraction** — `ExtractionService` detects format (`ProtocolFormat`) and runs the
   appropriate extractor, writing `text_blocks.parquet` and `tables.parquet`.
3. **ich_parser** — dual-path classification:
   - *Rule-based*: `IchParser` classifies text blocks against 16 ICH E6(R3) sections;
     low-confidence sections enter the `ReviewQueue`.
   - *Query-driven*: `QueryExtractor` runs a schema of structured queries per section,
     optionally transforming results via Claude Sonnet. `TemplateAssembler` merges
     query hits into an `AssembledProtocol` with per-section coverage reports.
4. **soa_extractor** — `SoaExtractor` parses the Schedule of Activities table from ICH
   sections (or from the query pipeline via `query_bridge`) and maps activities to
   CDISC USDM v4.0 entities (`UsdmTimepoint`, `UsdmActivity`, `UsdmScheduledInstance`).
5. **sdtm** — `SdtmService` generates SDTM trial design domains (TS, TA, TE, TV, TI)
   and `DefineXmlGenerator` produces Define-XML v2.1; `ValidationService` runs Pinnacle 21,
   FDA TCG, and Define-XML checks.
6. **pipeline** — `PipelineOrchestrator` orchestrates stages 1-5 with per-stage lineage
   checkpoints and SHA-256 chain verification.

### Key modules

| Module | Key files | Purpose |
|--------|-----------|---------|
| `ich_parser` | `query_extractor.py`, `template_assembler.py`, `query_schema.py` | Query-driven extraction with LLM transformation and protocol assembly |
| `ich_parser` | `parser.py`, `classifier.py`, `llm_retemplater.py` | Rule-based classification and LLM retemplating |
| `ich_parser` | `models.py`, `review_queue.py`, `parquet_writer.py` | Data models, human review queue, Parquet I/O |
| `soa_extractor` | `parser.py`, `extractor.py`, `query_bridge.py` | SoA table parsing with query pipeline integration |
| `soa_extractor` | `mapper.py`, `resolver.py`, `table_discovery.py` | USDM mapping, visit synonyms, PDF table discovery |
| `sdtm` | `domain_generators.py`, `define_xml.py`, `ct_normalizer.py` | SDTM domain generation and controlled terminology |
| `ui` | `app.py`, `components/query_pipeline.py` | Streamlit application and query pipeline UI |

## Requirements

- Python 3.11+
- Java runtime (JRE 8+) for `tabula-py`

### Python dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `minio` | >= 7.2.0 | MinIO client for WORM object storage (production) |
| `pyarrow` | >= 14.0.0 | Parquet read/write for all intermediate artifacts |
| `duckdb` | >= 0.10.0 | In-process analytics over Parquet files |
| `pdfplumber` | >= 0.11.0 | Primary PDF text and table extraction |
| `camelot-py[base]` | >= 0.11.0 | Lattice/stream table extraction fallback |
| `tabula-py` | >= 2.9.0 | Java-based table extraction (third fallback) |
| `pandas` | >= 2.0.0 | Tabular data manipulation |
| `lxml` | >= 5.0.0 | CTR-XML / CDISC ODM parsing |
| `PyYAML` | >= 6.0 | YAML schema loading |
| `requests` | >= 2.31.0 | HTTP client for registry API calls |
| `anthropic` | >= 0.40.0 | Claude API for LLM retemplating and query transformation |
| `pyreadstat` | >= 1.3.0 | SDTM XPT serialization |
| `streamlit` | >= 1.28.0 | Interactive web application |
| `plotly` | >= 5.0.0 | Schedule of Visits swimlane visualization |

## Installation

```bash
# Clone
git clone https://github.com/LittleHubOnThePrairie/Protocol-To-Cohort-Viz.git
cd Protocol-To-Cohort-Viz

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Streamlit application

```bash
streamlit run src/ptcv/ui/app.py
```

The application provides a tabbed interface:

| Tab | Description |
|-----|-------------|
| **Search** | Download protocols from ClinicalTrials.gov or EU-CTR |
| **Process** | Run ICH section classification and optional LLM retemplating |
| **Query Pipeline** | Run query-driven extraction with optional Claude Sonnet transformation |
| **SoA & SDTM** | Extract Schedule of Activities and generate SDTM datasets |
| **Benchmark** | View classification performance metrics |

### Run the full pipeline programmatically

```python
from ptcv.storage import FilesystemAdapter
from ptcv.pipeline import PipelineOrchestrator

gateway = FilesystemAdapter(root="data/protocols")
gateway.initialise()

orchestrator = PipelineOrchestrator(gateway=gateway)
result = orchestrator.run(pdf_bytes, registry_id="NCT12345678")

print(result.pipeline_run_id)
print(result.verify_lineage_chain())
```

### Download a protocol

```python
from ptcv.protocol_search import ClinicalTrialsService, FilestoreManager

filestore = FilestoreManager(root="data/protocols")
service = ClinicalTrialsService(filestore=filestore)
result = service.download(registry_id="NCT12345678")
```

### Query-driven extraction

```python
from ptcv.ich_parser.query_extractor import QueryExtractor

extractor = QueryExtractor(
    enable_transformation=True,
    anthropic_api_key="sk-...",
)
results = extractor.extract(protocol_text, registry_id="NCT12345678")
```

### Extract Schedule of Activities

```python
from ptcv.soa_extractor import SoaExtractor

extractor = SoaExtractor()
result = extractor.extract(
    sections=sections,
    registry_id="NCT12345678",
    source_run_id=parse_result.run_id,
    source_sha256=parse_result.artifact_sha256,
)
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Configuration

### Storage backends

| Backend | Class | Use case |
|---------|-------|---------|
| Local filesystem + SQLite | `FilesystemAdapter` | Development and testing |
| MinIO WORM + SQLite | `LocalStorageAdapter` | Production (GxP-compliant) |

```python
# Development (default)
from ptcv.storage import FilesystemAdapter
gateway = FilesystemAdapter(root="data/protocols")

# Production
from ptcv.storage import LocalStorageAdapter
gateway = LocalStorageAdapter(
    endpoint="minio.internal:9000",
    access_key="...",
    secret_key="...",
    bucket="protocols",
    db_path="data/lineage.db",
)
```

### Protocol data directory

Downloaded protocols are stored under `data/protocols/` by default:

```
data/protocols/
    clinicaltrials/      — ClinicalTrials.gov PDF/XML files (83 protocols)
    eu-ctr/              — EU-CTR / CTIS PDF/XML files
    metadata/            — Protocol metadata JSON
    lineage.db           — SQLite append-only lineage chain
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for LLM retemplating and query transformation |

## Testing

The test suite contains 2,180+ tests across all modules:

```
tests/
    ich_parser/         — ICH classification, query extraction, template assembly
    soa_extractor/      — SoA parsing, USDM mapping, query bridge
    sdtm/               — SDTM domain generation, Define-XML, validation
    extraction/         — PDF/XML extraction
    storage/            — Storage gateway adapters
    pipeline/           — End-to-end pipeline orchestration
    annotations/        — Span annotation service
    protocol_search/    — Protocol download and search
    ui/                 — Streamlit components and visualization
    smoke/              — Integration smoke tests
    scripts/            — Utility script tests
```

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific module
python -m pytest tests/ich_parser/ -v
python -m pytest tests/soa_extractor/ -v

# Run with coverage
python -m pytest tests/ --cov=ptcv --cov-report=term-missing
```

## Compliance

PTCV is designed for use in regulated clinical trial environments:

| Framework | Coverage |
|-----------|----------|
| **21 CFR Part 11** | Electronic records, audit trails, WORM storage, access controls |
| **ICH E6(R3)** | Appendix B section classification (16 sections) |
| **ALCOA+** | Attributable, Legible, Contemporaneous, Original, Accurate data integrity |
| **CDISC USDM v4.0** | Schedule of Activities entity model |
| **CDISC SDTM** | Trial design domains (TS, TA, TE, TV, TI) |
| **Define-XML v2.1** | Dataset metadata and controlled terminology |
| **NIST AI RMF** | Risk management for ML components |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for compliance requirements, code style, and
branch/commit conventions that apply to this regulated software project.

## Related

- Jira: [PTCV](https://littlehubonprairie.atlassian.net/jira/software/projects/PTCV/board)
- Confluence: [PTCV](https://littlehubonprairie.atlassian.net/wiki/spaces/PTCV)
