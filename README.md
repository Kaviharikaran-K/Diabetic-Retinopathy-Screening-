# Diabetic Retinopathy Screening

Explainable-AI (XAI) diabetic retinopathy screening system for **dual-eye fundus exams** (OD / OS). The project combines an offline computer-vision pipeline with a Streamlit clinical dashboard and a MATLAB SimEvents telemedicine simulator that consumes the exported risk-priority queue and latency profile.

## Features

- **Simultaneous dual-eye processing** on a worker thread pool – right eye (OD) and left eye (OS) in one pass.
- **Image Quality Assessment (IQA)** – sharpness, luminance, contrast and fundus coverage; flags ungradable eyes.
- **Pre-processing** – CLAHE enhancement on the green channel.
- **Anatomy & lesion segmentation** – optic disc, fovea, vessel network, microaneurysms, haemorrhages, hard exudates and neovascularisation.
- **ICDR classification (grade 0–4)** – INT8-quantised stacked neural ensemble (MobileViT-XS + DRNet13) fused with an evidence-based clinical staging profiler. Falls back gracefully to the clinical profiler when weights are absent, so the pipeline runs out of the box.
- **Explainability** – Grad-CAM heatmaps, attention rollout and a deterministic lesion/feature saliency overlay.
- **Human-in-the-loop review** – confirm / modify grade, mark ungradable, confirm referral, save doctor review.
- **Reporting & exports** – per-eye diagnostic report, risk-priority queue (JSON/CSV), clinical reviews (JSONL) and latency profile for the MATLAB simulator.
- **Fully offline** – no network calls at runtime.

| ICDR grade | Meaning | Priority |
|---|---|---|
| 0 | No apparent retinopathy | P3 – monitor in 6 months |
| 1 | Mild NPDR | P3 – monitor in 6 months |
| 2 | Moderate NPDR | P2 – routine referral |
| 3 | Severe NPDR | P1 – urgent (same week) |
| 4 | Proliferative DR | P0 – immediate specialist |

## Project structure

```
DR/
├── app_interface.py       # Streamlit clinical dashboard
├── pipeline_engine.py     # Core screening engine (IQA, segmentation, XAI, reporting)
├── telemedicine_sim.m     # MATLAB SimEvents telemedicine / queue simulator
├── requirements.txt       # Python dependencies
├── exports/               # Generated metrics, queues and review logs
├── .streamlit/            # Dashboard theme / server config
└── PROJECT_BUNDLE.txt     # Concatenated source bundle for quick reference
```

## Requirements

- Python 3.9+
- Hard dependencies: `numpy`, `opencv-python-headless`, `scipy`
- Dashboard: `streamlit`
- Optional neural features: `torch`, `torchvision`, `timm`, `scikit-image`

## Installation

```bash
git clone https://github.com/Kaviharikaran-K/Diabetic-Retinopathy-Screening-.git
cd Diabetic-Retinopathy-Screening-
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

To enable the neural ensemble, uncomment the optional packages in `requirements.txt` and install them:

```bash
pip install torch torchvision timm scikit-image
```

## Usage

### Streamlit dashboard

```bash
streamlit run app_interface.py
```

- Upload OD/OS fundus images, or use the **Synthetic demo** mode (no files required).
- Review the side-by-side raw fundus, heatmap, attention-rollout and lesion-annotation panels.
- Confirm or modify the AI grade, then save the doctor review.
- Use the MATLAB handoff panel to export metrics for the telemedicine simulator.

### Command line

```bash
# Real scans
python pipeline_engine.py --od OD.png --os OS.png --out exports

# Synthetic demo (no images needed)
python pipeline_engine.py --demo
python pipeline_engine.py --demo --od-severity proliferative --os-severity moderate
```

### From Python

```python
import pipeline_engine as pe

report = pe.run_case_paths(od_path="OD.png", os_path="OS.png")
pe.export_metrics(report, out_dir="exports")

for eye in (report.od, report.os):
    print(eye.eye, eye.icdr_grade, eye.confidence, eye.refer)
```

### MATLAB telemedicine simulation

Run `telemedicine_sim.m` in MATLAB/Simulink after generating `exports/` – it reads the risk-priority queue and latency profile produced by the screening engine.

## Outputs (`exports/`)

| File | Description |
|---|---|
| `batch_sim_metrics.json` | Per-eye and encounter-level metrics |
| `risk_priority_queue.csv` | Prioritised referral queue (P0–P3) |
| `clinical_reviews.jsonl` | Append-only log of clinician decisions |
| `latency_profile.csv` | Per-stage latency profile for the simulator |

## Optional model weights

Set `DR_WEIGHTS_PATH` to a trained PyTorch `state_dict` to enable the neural ensemble. Without it, the engine automatically uses the clinical ICDR staging profiler and deterministic feature saliency, so all features remain functional.

## Notes & limitations

- The engine is a research/prototype system and **not a certified medical device**. It does not replace clinical judgement.
- Neural-derived outputs (INT8 ensemble, Grad-CAM, attention rollout) require pre-trained weights; the clinical fallback is deterministic and heuristic.
- Synthetic demo images are for demonstration only and must not be used for clinical evaluation.

## License

No licence has been specified for this repository. Contact the author before redistribution or commercial use.
