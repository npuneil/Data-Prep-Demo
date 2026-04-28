# Data Preparation — On-Device AI Demo 📊

An on-device AI-powered data preparation application for financial services, running entirely on the NPU (Neural Processing Unit) via Microsoft Foundry Local. Scrapes, parses, extracts entities, maps to schema, and exports clean datasets — all locally. Optimized for Snapdragon X (QNN runtime).

## On-Device AI Prototypes & Sample Code

### Overview

This repository contains prototypes, demos, and sample code that illustrate patterns for building on-device AI solutions. The content is provided for educational and demonstration purposes only to help developers explore ideas and implementation approaches.

This repository does not contain Microsoft products and is not a supported or production-ready offering.

### Prototype & Sample Code Disclosure

- All code and demos are experimental prototypes or samples.
- They may be incomplete, change without notice, or be removed at any time.
- The contents are provided "as-is," without warranties or guarantees of any kind.

### No Product, Performance, or Business Claims

- This repository makes no claims about performance, accuracy, productivity, efficiency, cost savings, reliability, or security.
- Any example outputs, screenshots, or logs are illustrative only and should not be interpreted as typical or expected results.

### AI Output Variability

- AI and machine-learning outputs may be non-deterministic, incomplete, or incorrect.
- Example outputs shown here are not guaranteed and may vary across runs, devices, or environments.

### Responsible AI Considerations

- These samples are intended to demonstrate technical patterns, not validated AI systems.
- Developers are responsible for evaluating fairness, reliability, privacy, accessibility, and safety before using similar approaches in real applications.
- Do not deploy AI solutions based on this code without appropriate testing, human oversight, and safeguards.

### Data & Fictitious Content

- Any names, data, or scenarios used in examples are fictitious and for illustration only.
- Do not use real personal, customer, or confidential data without proper authorization and protections.

### Third-Party Components

- The repository may reference third-party libraries or tools.
- Use of those components is subject to their respective licenses and terms.

### No Support

Microsoft does not provide support, SLAs, or warranties for the contents of this repository.

### Summary

By using this repository, you acknowledge that it contains illustrative prototypes and sample code only, not supported or production-ready software.

---

## Quick Start

```powershell
# First time — install dependencies:
launch.bat         # or: .\launch.ps1

# Starts Streamlit at http://localhost:8501
```

## Prerequisites

- **Windows 11 Copilot+ PC** with Snapdragon X NPU
- **Python 3.10+** (ARM64-native recommended for Snapdragon)
- **Foundry Local** installed (`winget install Microsoft.FoundryLocal`)

## Snapdragon X Optimization

This app is optimized for ARM64 Snapdragon X devices:
- **ONNX Runtime** with NPU acceleration
- **Local-only processing** — sensitive financial data never leaves the device
- **CPU fallback** when no NPU model is available

## Features

| Step | Description |
|------|-------------|
| **1. Data Capture** | Scrape websites or parse PDF documents |
| **2. Entity Extraction** | NPU extracts organizations, currencies, dates, and financial terms |
| **3. Schema Mapping** | Map extracted entities to target schema with confidence scores |
| **4. Export** | Clean, structured datasets in CSV or JSON format |

## Architecture

- **Frontend**: Streamlit
- **NPU Engine**: ONNX Runtime with Foundry Local
- **Modules**: Scraper, PDF Parser, Entity Extractor, Normalizer, Schema Mapper, Exporter, Provenance Tracker

## Demo Experience

See `DEMO_SCRIPT.md` for a guided demo walkthrough.

**The key demo moment:** Walk through the 4-step pipeline from raw unstructured data to clean structured output. All entity extraction and schema mapping happens on-device via NPU — no cloud APIs, no data exfiltration risk.
