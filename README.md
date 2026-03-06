# Protocol-To-Cohort-Viz (PTCV)

> ML-enabled clinical trial software for parsing, classifying, and transforming clinical trial
> protocols into CDISC-compliant SDTM datasets.

## Overview

PTCV ingests clinical trial protocols from public registries (EU-CTR / CTIS and ClinicalTrials.gov),
extracts structured content from PDF and CTR-XML documents, classifies sections against the
ICH E6(R3) Appendix B format, builds a CDISC USDM v4.0 Schedule of Activities model, and
generates SDTM trial design datasets (TS, TA, TE, TV, TI) with accompanying Define-XML v2.1.
All artifact writes are immutable (WORM) with a tamper-evident SHA-256 lineage chain, satisfying
21 CFR Part 11, ICH E6(R3), and ALCOA+ requirements.

## Architecture

```
src/ptcv/
    protocol_search/    — Search and download protocols from EU-CTR and ClinicalTrials.gov
    extraction/         — Format detection and text/table extraction from PDF and CTR-XML
    ich_parser/         — ICH E6(R3) section classifier (rule-based and RAG backends)
    soa_extractor/      — Schedule of Activities parser and CDISC USDM v4.0 mapper
    sdtm/               — SDTM domain generators, Define-XML, CT normalizer, and validator
    storage/            — StorageGateway abstraction (FilesystemAdapter / MinIO WORM)
    compliance/         — Audit trail, data integrity, and access control (21 CFR Part 11)
    pipeline/           — End-to-end PipelineOrchestrator wiring all 6 stages
```

Data flows through the pipeline in order:

1. **protocol_search** — retrieves protocol PDFs or CTR-XML from registries, stores via
   `StorageGateway` with SHA-256 checksums.
2. **extraction** — `ExtractionService` detects format (`ProtocolFormat`) and runs the
   appropriate extractor, writing `text_blocks.parquet` and `tables.parquet`.
3. **ich_parser** — `IchParser` classifies text blocks against 11 ICH E6(R3) sections and
   writes classified sections to Parquet; low-confidence sections enter the `ReviewQueue`.
4. **soa_extractor** — `SoaExtractor` parses the Schedule of Activities table and maps
   activities to CDISC USDM v4.0 entities.
5. **sdtm** — `SdtmService` generates SDTM trial design domains and `DefineXmlGenerator`
   produces Define-XML v2.1; `ValidationService` runs Pinnacle 21, FDA TCG, and Define-XML checks.
6. **pipeline** — `PipelineOrchestrator` orchestrates stages 1–5 with per-stage lineage
   checkpoints and SHA-256 chain verification.

## Requirements

- Python 3.11+
- `minio>=7.2.0` — MinIO client for WORM object storage (production)
- `pyarrow>=14.0.0` — Parquet read/write for all intermediate artifacts
- `duckdb>=0.10.0` — In-process analytics over Parquet files
- `pdfplumber>=0.11.0` — Primary PDF text and table extraction
- `camelot-py[base]>=0.11.0` — Lattice/stream table extraction fallback
- `tabula-py>=2.9.0` — Java-based table extraction (third fallback)
- `pandas>=2.0.0` — Tabular data manipulation
- `lxml>=5.0.0` — CTR-XML / CDISC ODM parsing
- `requests>=2.31.0` — HTTP client for registry API calls

## Installation

```bash
# Clone
git clone https://github.com/LittleHubOnThePrairie/Protocol-To-Cohort-Viz.git
cd Protocol-To-Cohort-Viz

# Install dependencies
pip install -r requirements.txt
```

> `tabula-py` requires a Java runtime (JRE 8+). Install OpenJDK if not already present.

## Usage

### Run the full pipeline

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

### Classify ICH sections

```python
from ptcv.ich_parser import IchParser, RuleBasedClassifier

parser = IchParser(classifier=RuleBasedClassifier())
parse_result = parser.parse(protocol_text, registry_id="NCT12345678")
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
    clinicaltrials/      — ClinicalTrials.gov PDF/XML files
    eu-ctr/              — EU-CTR / CTIS PDF/XML files
    metadata/            — Protocol metadata JSON
    lineage.db           — SQLite append-only lineage chain
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for compliance requirements, code style, and
branch/commit conventions that apply to this regulated software project.

## Related

- Jira: [PTCV](https://littlehubonprairie.atlassian.net/jira/software/projects/PTCV/board)
- Confluence: [PTCV](https://littlehubonprairie.atlassian.net/wiki/spaces/PTCV)
- Changelog: [PTCV/jira_changelog.md](PTCV/jira_changelog.md)
