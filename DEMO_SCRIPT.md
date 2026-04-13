# Data Prep Assistant — 90-Second Executive Demo Script

> **Presenter**: Have the app open at `http://localhost:8512` before starting.
> **Setup**: Make sure no data is loaded yet (fresh session — restart the app if needed).

---

## Opening (0:00 – 0:10)

> "This is our on-device AI data prep tool running entirely on this Surface CoPilot+ PC. No cloud, no API calls — everything stays on-device. Let me show you the full pipeline."

**Point out**: Sidebar shows **NPU Active — QNN Hexagon**, the green pulsing dot, and the **100% On-Device** privacy badge at the bottom.

---

## Step 1 — Data Capture (0:10 – 0:25)

1. Click the orange **🚀 Load Demo PDF (Zava Q1 2026)** button at the top.
2. The PDF parses instantly — point out the page count, word count, and table count in the preview metrics.
3. Briefly scroll the text preview.

> "We just ingested a Zava quarterly portfolio review — parsed locally in milliseconds. You can also capture live web pages or upload any PDF."

---

## Step 2 — Extract & Normalize (0:25 – 0:50)

1. Click the **🔍 Extract & Normalize** tab.
2. Click **🔍 Run Entity Extraction**.
3. Watch the progress bar — call out **"Running ONNX inference on QNN Hexagon NPU"** as it processes.
4. When complete, point out:
   - **Entities Found** count (tickers, ISINs, NAV values, returns, risk metrics)
   - **Entity Distribution** bar chart
   - **Confidence Distribution** histogram
   - **Data Quality scores** at the bottom (Overall Quality, Completeness, Accuracy)

> "The NPU just ran our ML model on the Qualcomm Hexagon — extracting financial entities like tickers, ISINs, returns, and risk metrics. All AI inference happened right here on the chip."

---

## Step 3 — Schema Mapping (0:50 – 1:05)

1. Click the **🗺️ Schema Mapping** tab.
2. Click **🗺️ Run Schema Mapping**.
3. Point out the **Schema Coverage** percentage and the coverage-by-category bar chart.

> "Extracted data is now mapped to our security master schema — ready for downstream systems. No manual field mapping needed."

---

## Step 4 — Export (1:05 – 1:15)

1. Click the **📦 Export** tab.
2. Show the **JSON**, **CSV**, and **Provenance** download buttons.
3. Expand the JSON preview to show the structured output.

> "One click to export — JSON, CSV, or full data lineage. Every transformation is tracked for compliance."

---

## Closing — Why This Matters (1:15 – 1:30)

Point to the **sidebar** sustainability metrics:

> "Every inference we just ran saved carbon and cost versus cloud APIs. At scale — say 5,000 inferences a month — that's real dollars and real carbon reduction. And most importantly: **proprietary financial data never left this device**. No cloud, no latency, full compliance."

**Key talking points if asked**:
- **Privacy**: Data never leaves the machine — critical for proprietary financial data
- **Cost**: Zero cloud API costs — hardware does all the work
- **Compliance**: Meets data residency and sovereignty requirements
- **Offline**: Works on planes, in secure facilities, anywhere
- **NPU advantage**: Dedicated AI silicon offloads ML work from the CPU — both run simultaneously

---

## If Time Permits — NPU Benchmark (bonus)

1. Click the **⚡ NPU Benchmark** tab.
2. Leave iterations at 100, click **🏃 Run Benchmark**.
3. Show the CPU vs NPU comparison cards and throughput chart.

> "This is a real head-to-head benchmark — CPU versus Qualcomm Hexagon NPU, measured live on this device."

---

## Troubleshooting

| Issue | Fix |
|---|---|
| App won't start | `cd C:\ZavaDataPrep` then `C:\ZavaDataPrep\.venv\Scripts\streamlit.exe run app.py --server.port 8512` |
| Blank screen | Hard-refresh browser (Ctrl+Shift+R) |
| NPU shows "Not Detected" | Restart app — QNN session needs ARM64 Python and onnxruntime-qnn |
| Sidebar shows 0 inferences | Normal on fresh load — numbers update after extraction or benchmark |
