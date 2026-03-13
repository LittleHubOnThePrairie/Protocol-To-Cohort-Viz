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

The pipeline employs a **2D graceful degradation matrix** with independent extraction and
classification axes, ensuring the system always produces output even when optional components
(Docling, NeoBERT, Claude API) are unavailable.

The project ships a Streamlit application with two independent processing pipelines:

- **Process (LLM Retemplater)** — rule-based ICH section classification followed by optional
  LLM-powered retemplating against the full ICH E6(R3) Appendix B structure.
- **Query Pipeline** — query-driven extraction using a pre-defined schema of questions per
  section, with optional Claude Sonnet transformation for content reframing.

Both pipelines feed the SoA extraction and SDTM generation stages.

## Architecture

```
src/ptcv/
    protocol_search/    -- Search and download protocols from EU-CTR and ClinicalTrials.gov
    extraction/         -- Format detection, text/table extraction from PDF and CTR-XML,
                           and hybrid Claude vision enhancement for sparse pages (PTCV-172)
    ich_parser/         -- ICH E6(R3) section classifier, LLM retemplater, query extractor,
                           classification cascade (NeoBERT + RAG + Sonnet), and template
                           assembler
    soa_extractor/      -- Schedule of Activities parser with 3-level cascade (table, vision,
                           LLM text construction), CDISC USDM v4.0 mapper, and query bridge
    sdtm/               -- SDTM domain generators, Define-XML, CT normalizer, and validator
    annotations/        -- Span-level annotation service for section boundaries
    storage/            -- StorageGateway abstraction (FilesystemAdapter / MinIO WORM)
    compliance/         -- Audit trail, data integrity, and access control (21 CFR Part 11)
    pipeline/           -- PipelineOrchestrator with 2D degradation chain (PTCV-163)
    analysis/           -- Batch analysis runner and quality benchmarking
    benchmark/          -- Extraction quality comparator and metrics
    ui/                 -- Streamlit application with modular component library
```

### Pipeline stages

1. **protocol_search** -- retrieves protocol PDFs or CTR-XML from registries, stores via
   `StorageGateway` with SHA-256 checksums.
2. **extraction** -- `ExtractionService` detects format (`ProtocolFormat`) and runs the
   appropriate extractor, writing `text_blocks.parquet` and `tables.parquet`. At E1 level,
   `VisionEnhancer` renders sparse/low-quality pages to PNG and sends them to Claude vision
   API for structured JSON extraction.
3. **soa_extraction** -- `SoaExtractor` runs a 3-level cascade to build USDM entities:
   - *Level 1*: Table-based extraction (pre-extracted tables, PDF discovery, text parsing)
   - *Level 2*: Claude vision extraction for complex merged-cell tables (PTCV-172)
   - *Level 3*: Sonnet text construction from B.4/B.7/B.8/B.9 prose (PTCV-166)
4. **classification_cascade** -- `ClassificationRouter` runs a hybrid local + Sonnet cascade:
   rule-based or NeoBERT classifiers produce initial predictions; low-confidence sections are
   routed to Claude Sonnet with optional RAG context from prior protocols (PTCV-161, PTCV-162).
5. **retemplating** -- `LlmRetemplater` reorganizes non-ICH protocols into standard
   ICH E6(R3) structure via two-pass Claude Opus classification and text assembly.
6. **sdtm** -- `SdtmService` generates SDTM trial design domains (TS, TA, TE, TV, TI)
   and `DefineXmlGenerator` produces Define-XML v2.1; `ValidationService` runs Pinnacle 21,
   FDA TCG, and Define-XML checks.
7. **pipeline** -- `PipelineOrchestrator` orchestrates stages 1-6 with per-stage lineage
   checkpoints and SHA-256 chain verification.

### 2D degradation matrix (PTCV-163)

The pipeline selects the highest available quality level on each axis independently:

| Extraction | Description | Requirements |
|-----------|-------------|--------------|
| E1 | Docling + Vision | `docling`, `PTCV_VISION_API_KEY` |
| E2 | Docling only | `docling` |
| E3 | pdfplumber cascade | None (always available) |

| Classification | Description | Requirements |
|---------------|-------------|--------------|
| C1 | NeoBERT + RAG + Sonnet | `torch`, `transformers`, NeoBERT model, RAG index, `ANTHROPIC_API_KEY` |
| C2 | NeoBERT + Sonnet | `torch`, `transformers`, NeoBERT model, `ANTHROPIC_API_KEY` |
| C3 | RuleBased + Sonnet | `ANTHROPIC_API_KEY` |
| C4 | NeoBERT + RuleBased | `torch`, `transformers`, NeoBERT model |
| C5 | RuleBased only | None (always available) |

E3+C5 is guaranteed to work with zero external dependencies.

### SoA extraction cascade (PTCV-166)

| Level | Source | Method | Trigger |
|-------|--------|--------|---------|
| 1 | Pre-extracted tables / PDF discovery / text parsing | `SoaTableParser`, `TableDiscovery` | Always tried first |
| 2 | Claude vision API | `VisionEnhancer` | E1 extraction level |
| 3 | Sonnet text construction | `LlmSoaBuilder` | < 3 activities from L1/L2 |

### Key modules

| Module | Key files | Purpose |
|--------|-----------|---------|
| `extraction` | `extraction_service.py`, `pdf_extractor.py`, `vision_enhancer.py` | PDF/XML extraction with vision fallback |
| `ich_parser` | `classification_router.py`, `neobert_classifier.py`, `rag_index.py` | Hybrid classification cascade with ML and RAG |
| `ich_parser` | `query_extractor.py`, `template_assembler.py`, `query_schema.py` | Query-driven extraction with LLM transformation |
| `ich_parser` | `parser.py`, `classifier.py`, `llm_retemplater.py` | Rule-based classification and LLM retemplating |
| `soa_extractor` | `extractor.py`, `llm_soa_builder.py`, `query_bridge.py` | 3-level SoA cascade with LLM text construction |
| `soa_extractor` | `mapper.py`, `resolver.py`, `table_discovery.py` | USDM mapping, visit synonyms, PDF table discovery |
| `sdtm` | `domain_generators.py`, `define_xml.py`, `ct_normalizer.py` | SDTM domain generation and controlled terminology |
| `pipeline` | `orchestrator.py`, `degradation.py` | Pipeline orchestration with 2D degradation chain |
| `ui` | `app.py`, `components/query_pipeline.py` | Streamlit application and query pipeline UI |

## Requirements

- Python 3.13+ (CUDA torch compatibility)
- Java runtime (JRE 8+) for `tabula-py`

### Python dependencies (runtime)

| Package | Version | Purpose |
|---------|---------|---------|
| `minio` | >= 7.2.0 | MinIO client for WORM object storage (production) |
| `pyarrow` | >= 14.0.0 | Parquet read/write for all intermediate artifacts |
| `duckdb` | >= 0.10.0 | In-process analytics over Parquet files |
| `PyMuPDF` | >= 1.24.0 | PDF rendering for vision extraction |
| `pymupdf4llm` | >= 0.0.17 | PDF-to-markdown conversion |
| `pdfplumber` | >= 0.11.0 | PDF text and table extraction |
| `camelot-py[base]` | >= 0.11.0 | Lattice/stream table extraction fallback |
| `tabula-py` | >= 2.9.0 | Java-based table extraction (third fallback) |
| `pandas` | >= 2.0.0 | Tabular data manipulation |
| `spacy` | >= 3.8.0 | NLP for section matching |
| `lxml` | >= 5.0.0 | CTR-XML / CDISC ODM parsing |
| `PyYAML` | >= 6.0 | YAML schema loading |
| `requests` | >= 2.31.0 | HTTP client for registry API calls |
| `anthropic` | >= 0.40.0 | Claude API for classification, retemplating, vision, and SoA construction |
| `pyreadstat` | >= 1.3.0 | SDTM XPT serialization |
| `streamlit` | >= 1.28.0 | Interactive web application |
| `plotly` | >= 5.0.0 | Schedule of Visits swimlane visualization |

### ML dependencies (optional)

Install separately for NeoBERT fine-tuning and RAG index support:

```bash
pip install -r requirements-ml.txt
```

| Package | Purpose |
|---------|---------|
| `torch` >= 2.10.0 | PyTorch for NeoBERT |
| `transformers` >= 4.38.0 | Hugging Face model loading |
| `sentence-transformers` >= 2.2.0 | RAG embedding model |
| `faiss-cpu` >= 1.7.4 | Vector similarity search for RAG index |

## Installation

```bash
# Clone
git clone https://github.com/LittleHubOnThePrairie/Protocol-To-Cohort-Viz.git
cd Protocol-To-Cohort-Viz

# Install runtime dependencies
pip install -r requirements.txt

# (Optional) Install ML dependencies for NeoBERT/RAG
pip install -r requirements-ml.txt
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
| **Analysis** | Batch analysis runner and extraction quality benchmarking |

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
    clinicaltrials/      -- ClinicalTrials.gov PDF/XML files (83 protocols)
    eu-ctr/              -- EU-CTR / CTIS PDF/XML files
    metadata/            -- Protocol metadata JSON
    lineage.db           -- SQLite append-only lineage chain
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for classification cascade, retemplating, vision, and SoA construction |
| `PTCV_VISION_API_KEY` | API key for Claude vision extraction (enables E1 degradation level) |
| `PTCV_VISION_MAX_PAGES` | Max pages for vision extraction per protocol (default: 5) |
| `PTCV_NEOBERT_MODEL` | Path to NeoBERT checkpoint directory (enables C1/C2/C4 classification) |
| `PTCV_RAG_INDEX` | Path to RAG vector index directory (enables C1 classification) |

## Testing

The test suite contains 3,200+ tests across all modules:

```
tests/
    ich_parser/         -- ICH classification, query extraction, template assembly,
                           classification cascade, NeoBERT, RAG index
    soa_extractor/      -- SoA parsing, USDM mapping, query bridge, LLM SoA builder
    sdtm/               -- SDTM domain generation, Define-XML, validation
    extraction/         -- PDF/XML extraction, vision enhancer, markdown normalizer
    storage/            -- Storage gateway adapters
    pipeline/           -- Pipeline orchestration, degradation chain
    analysis/           -- Batch analysis, benchmarking
    annotations/        -- Span annotation service
    protocol_search/    -- Protocol download and search
    ui/                 -- Streamlit components and visualization
    smoke/              -- Integration smoke tests
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
