# On-Device AI Prototypes & Sample Code

## Overview

This repository contains prototypes, demos, and sample code that illustrate patterns for building on-device AI solutions. The content is provided for educational and demonstration purposes only to help developers explore ideas and implementation approaches.

This repository does not contain Microsoft products and is not a supported or production-ready offering.

## Prototype & Sample Code Disclosure

- All code and demos are experimental prototypes or samples.
- They may be incomplete, change without notice, or be removed at any time.
- The contents are provided "as-is," without warranties or guarantees of any kind.

## No Product, Performance, or Business Claims

- This repository makes no claims about performance, accuracy, productivity, efficiency, cost savings, reliability, or security.
- Any example outputs, screenshots, or logs are illustrative only and should not be interpreted as typical or expected results.

## AI Output Variability

- AI and machine-learning outputs may be non-deterministic, incomplete, or incorrect.
- Example outputs shown here are not guaranteed and may vary across runs, devices, or environments.

## Responsible AI Considerations

- These samples are intended to demonstrate technical patterns, not validated AI systems.
- Developers are responsible for evaluating fairness, reliability, privacy, accessibility, and safety before using similar approaches in real applications.
- Do not deploy AI solutions based on this code without appropriate testing, human oversight, and safeguards.

## Data & Fictitious Content

- Any names, data, or scenarios used in examples are fictitious and for illustration only.
- Do not use real personal, customer, or confidential data without proper authorization and protections.

## Third-Party Components

- The repository may reference third-party libraries or tools.
- Use of those components is subject to their respective licenses and terms.

## No Support

Microsoft does not provide support, SLAs, or warranties for the contents of this repository.

---

# ⭐ Data Prep Assistant

> **This is a demo / prototype / reference implementation.**

**On-Device AI-Powered Financial Data Preparation**

> A polished demo running on Microsoft Surface CoPilot+ PC showcasing how on-device AI with NPU acceleration can capture, extract, normalize, and export financial data — with complete data provenance and zero cloud dependency.

---

## 🚀 Quick Start

### Option 1: One-Click Launch (Recommended)
```powershell
# Right-click launch.ps1 → "Run with PowerShell"
# Or from terminal:
cd C:\ZavaDataPrep
powershell -ExecutionPolicy Bypass -File launch.ps1
```

### Option 2: Double-Click
Simply double-click **`launch.bat`** to start.

### Option 3: Manual Setup
```powershell
cd C:\ZavaDataPrep
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

## 📋 Demo Walkthrough

### Step 1: Data Capture
- Enter a public URL or use the built-in demo data (Vanguard Fund Report)
- Alternatively, upload a PDF fund fact sheet
- All fetching and parsing happens locally on-device

### Step 2: Entity Extraction & Normalization
- AI-powered extraction of financial entities (tickers, ISINs, NAVs, returns, risk metrics, etc.)
- Pattern-based + ML extraction with confidence scoring
- Automatic data normalization and quality assessment

### Step 3: Schema Mapping
- Maps extracted entities to the Security Master schema (v3.2.1)
- Visual coverage analysis across 12 data categories
- Validation against schema constraints

### Step 4: Export
- Download schema-aligned JSON or CSV exports
- Full data provenance chain attached
- Ready for direct ingestion into target systems

### NPU Benchmark
- Compare CPU vs NPU processing performance
- Shows 3-5x speedup with CoPilot+ PC NPU
- Demonstrates DirectML execution provider

---

## 🔒 Privacy & Security

| Feature | Detail |
|---------|--------|
| **Processing** | 100% on-device — no data leaves the machine |
| **Network** | Only used for initial web page fetch (optional) |
| **Models** | All AI models run locally via ONNX Runtime |
| **Storage** | Exports saved locally to `./output/` |
| **Compliance** | Full provenance chain for audit trail |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Web UI                     │
├───────────┬───────────┬───────────┬─────────────────┤
│  Data     │  Entity   │  Schema   │   Export &      │
│  Capture  │  Extract  │  Mapping  │   Provenance    │
├───────────┴───────────┴───────────┴─────────────────┤
│              ONNX Runtime + DirectML                  │
│         (NPU Acceleration on CoPilot+ PC)            │
├─────────────────────────────────────────────────────┤
│           Microsoft Surface CoPilot+ PC              │
│         Snapdragon X Elite · Hexagon NPU             │
└─────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
ZavaDataPrep/
├── app.py                    # Main Streamlit application
├── launch.ps1                # PowerShell one-click launcher
├── launch.bat                # Batch file launcher
├── requirements.txt          # Python dependencies
├── config/
│   ├── schema.json              # Target schema definition
│   └── settings.yaml            # Application settings
├── src/
│   ├── npu_engine.py         # NPU detection & acceleration
│   ├── scraper.py            # Web page capture & parsing
│   ├── pdf_parser.py         # PDF document processing
│   ├── entity_extractor.py   # Financial entity extraction
│   ├── normalizer.py         # Data normalization & quality
│   ├── schema_mapper.py      # Target schema mapping
│   ├── exporter.py           # JSON/CSV export generation
│   └── provenance.py         # Data lineage tracking
├── assets/
│   └── styles.css            # Custom UI styling
├── output/                   # Export destination
└── .streamlit/
    └── config.toml           # Streamlit theme config
```

---

## 🛠️ Technology Stack

- **Python 3.10+** — Core runtime
- **Streamlit** — Interactive web UI
- **ONNX Runtime + DirectML** — NPU-accelerated AI inference
- **BeautifulSoup4** — Web page parsing
- **PyMuPDF** — PDF processing
- **Pandas** — Data manipulation
- **Plotly** — Interactive visualizations

---

## 💡 Key Value Propositions

1. **Privacy-First**: All processing on-device — critical for proprietary financial data
2. **NPU Acceleration**: 3-5x faster AI inference on CoPilot+ PC hardware
3. **Offline Capable**: No internet required after initial data capture
4. **Cost Efficient**: Zero cloud API costs for AI processing
5. **Schema-Ready**: Direct alignment with target ingestion format
6. **Auditable**: Complete data provenance from source to output
7. **Energy Efficient**: NPU uses fraction of CPU/GPU power for AI workloads

---

*Built for Microsoft Surface CoPilot+ PC · Powered by ONNX Runtime DirectML*
