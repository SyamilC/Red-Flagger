# Red-Flagger

Dating Red Flag Detector: an interactive profile auditing prototype for multi-label dating profile risk signals.

The project combines text, tabular, and visual signals from the OkCupid dataset plus synthetic dating-bio examples. The final notebook builds an auditor-style report with Precision, Recall, and visual IoU context. The Chrome extension folder is intended for the bonus browser-side demo using TensorFlow.js.

## Assignment Mapping

| Requirement | Current project area |
| --- | --- |
| OkCupid dataset plus synthetic modern dating bios | `data/`, `notebooks/01_data_exploration.ipynb`, `notebooks/02_nlp_model_training.ipynb` |
| Multi-label class imbalance handling | `notebooks/02_nlp_model_training.ipynb`, `notebooks/03_tabular_smote_model.ipynb` |
| Sarcasm and textual tone labels | NLP taxonomy and keyword rules in Notebook 02 |
| YOLOv8 visual analysis and IoU | `notebooks/04_yolo_visual_module.ipynb`, `reports/yolo_visual_metrics.csv` |
| Interactive auditing agent and reports | `notebooks/05_agent_inference_demo.ipynb`, `reports/` |
| Conceptual mobile interface | HTML phone mockup rendered in Notebook 05 |
| Chrome extension bonus | `chrome_extension/` |
| TensorFlow.js browser model export | Starter helpers in `scripts/` and target asset notes in `chrome_extension/models/` |

OpenRouter is treated as the course-recommended local/source LLM interface for the report beautification layer.

## Repository Layout

```text
api/                 Optional API surface
chrome_extension/    Browser extension prototype and TensorFlow.js target assets
data/                OkCupid-derived dataset and visual sample data
models/              Trained sklearn, DistilBERT, tabular, and YOLO artifacts
notebooks/           Main project workflow, from data prep to final auditor demo
reports/             Generated metrics, JSON assessments, and Markdown reports
scripts/             Utility scripts for browser/TensorFlow.js export work
```

## Environment Setup

Conda setup:

```bash
conda env create -f environment.yml
conda activate red-flagger
```

Pip setup inside an existing Python environment:

```bash
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

Run the notebooks in order when regenerating the full pipeline:

```text
01_data_exploration.ipynb
02_nlp_model_training.ipynb
03_tabular_smote_model.ipynb
04_yolo_visual_module.ipynb
05_agent_inference_demo.ipynb
```

## Chrome Extension Check

Static checks from the project root:

```bash
node --check chrome_extension/content.js
node --check chrome_extension/popup.js
node --check chrome_extension/inference.js
```

Manifest and asset check in PowerShell:

```powershell
Get-Content -Raw chrome_extension/manifest.json | ConvertFrom-Json | Out-Null
Test-Path chrome_extension/popup.html
Test-Path chrome_extension/content.js
Test-Path chrome_extension/inference.js
Test-Path chrome_extension/models/nlp_model.json
Test-Path chrome_extension/models/tabular_model_tfjs/model.json
Test-Path chrome_extension/models/yolo_model_tfjs/model.json
```

Manual Chrome check:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked and select `chrome_extension/`.
4. If Chrome shows a manifest or missing-file error, fix that first.
5. Open a page that contains profile-like DOM fields.
6. Click the extension icon and press Analyze Current Profile.
7. Right-click the popup and choose Inspect to view popup console errors.
8. Open the target page DevTools console to view content-script errors.

A working extension should load without manifest errors, show a ready status after model loading, extract profile text from the page, and render either red-flag scores or a no-major-flags result.

## TensorFlow.js Export Starters

The extension-side model assets should live under `chrome_extension/models/`:

```text
chrome_extension/models/nlp_model.json
chrome_extension/models/tabular_model_tfjs/model.json
chrome_extension/models/yolo_model_tfjs/model.json
```

Export the TF-IDF NLP model metadata for browser-side scoring:

```bash
python scripts/export_nlp_for_tfjs.py
```

Export YOLO weights to TensorFlow.js format when the installed Ultralytics/TensorFlow.js toolchain supports it:

```bash
python scripts/export_yolo_for_tfjs.py
```

The tabular SMOTE Random Forest model is not directly a TensorFlow.js LayersModel. For a browser demo, either export a simplified JSON scorer or train a small neural tabular surrogate before loading it with `tf.loadLayersModel`.

## Current Reports

Model metrics and profile audits are written to `reports/`. The final notebook also saves JSON assessments alongside Markdown reports so the report text can be traced back to structured model outputs.
