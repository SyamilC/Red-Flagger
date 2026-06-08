# Red-Flagger

Dating Red Flag Detector is an interactive profile auditing prototype for multi-label dating profile risk signals. The main demo is the local FastAPI mobile-style UI, which combines bio text, OkCupid-style metadata, optional profile photo upload, YOLO visual analysis, and an explainable audit report.

The Chrome extension is included as a bonus browser-side demo. For grading, walkthroughs, and most testing, start with the local UI.

## What The App Does

- Audits dating bios for sarcasm, cynicism, controlling language, negativity, entitlement, hookup focus, substance risk, and related labels.
- Uses OkCupid-style profile metadata such as age, body type, education, children/offspring, smoking, drinking, drugs, job, pets, religion, sign, and more.
- Handles imbalanced tabular classification through the SMOTE-trained tabular model.
- Accepts an optional profile photo and runs YOLOv8 automatically as part of the same full audit.
- Shows Precision, Recall, and visual IoU metrics from the generated reports.
- Produces an end-user audit report using either the local fallback writer or OpenRouter when a key is provided.

## Quick Start

Clone or download the project, then open a terminal in the project root folder:

```text
Red-Flagger/
```

All commands below assume your terminal is already inside that folder.

## Environment Setup

Python 3.11 is recommended. You can use either Conda or a normal virtual environment.

### Option A: Conda

```powershell
conda env create -f environment.yml
conda activate red-flagger
```

If `conda` is not recognized on Windows, open Anaconda Prompt, or add Conda to your terminal PATH, then run the same commands again.

### Option B: Python venv

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Useful main UI dependency check:

```powershell
python -c "import fastapi, multipart, ultralytics; print('main UI deps ok')"
```

The optional transformer module needs Hugging Face Transformers. If the UI says the transformer is unavailable, install:

```powershell
python -m pip install transformers accelerate safetensors
```

TensorFlow and TensorFlow.js Python packages are mainly needed for model export/conversion work. The Chrome extension runtime uses the npm TensorFlow.js package inside `chrome_extension/`.

Optional TensorFlow export/conversion check:

```powershell
python -c "import tensorflow as tf; import tensorflowjs as tfjs; print('tensorflow export deps ok')"
```

## Run The Mobile Auditor UI

Start the local API from the project root:

```powershell
python -m uvicorn api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/
```

Use the UI:

1. Fill in the bio and profile metadata.
2. Optionally attach a profile photo.
3. Optionally paste an OpenRouter key into the Evidence Console `Beautify key` field.
4. Click `Run Full Audit`.

If a photo is attached, the app automatically runs text, tabular, and YOLO visual analysis in one audit. If no photo is attached, it runs text and tabular analysis with the selected visual preset.

## OpenRouter And Report Source

The models still run either way. The report source badge only describes who wrote the final prose explanation.

- `local_fallback`: the app used the built-in local report writer because OpenRouter was not requested, no key was available, or the OpenRouter request failed.
- `openrouter`: the app successfully used OpenRouter to beautify the final report.

You can provide an OpenRouter key in either place:

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
python -m uvicorn api.app:app --reload
```

macOS/Linux:

```bash
export OPENROUTER_API_KEY="your_key_here"
python -m uvicorn api.app:app --reload
```

You can also paste the key directly into the UI before running an audit.

## API Checks

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Sample audit:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/audit/sample/0
```

Photo upload audit from PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/audit/photo `
  -F "bio=Fluent in sarcasm and always right." `
  -F "profile_id=demo_photo" `
  -F "tabular_json={}" `
  -F "use_openrouter=false" `
  -F "photo=@data/test_images/profile_000.jpg;type=image/jpeg"
```

Photo upload audit from macOS/Linux:

```bash
curl -X POST http://127.0.0.1:8000/audit/photo \
  -F "bio=Fluent in sarcasm and always right." \
  -F "profile_id=demo_photo" \
  -F "tabular_json={}" \
  -F "use_openrouter=false" \
  -F "photo=@data/test_images/profile_000.jpg;type=image/jpeg"
```

## Chrome Extension Bonus

The browser extension lives in `chrome_extension/`. It is the bonus path, not the main UI demo.

Build and check it:

```powershell
cd chrome_extension
npm install
npm run check
npm run build
```

Load it in Chrome:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select the `chrome_extension/` folder inside this project.
5. Open a profile-like web page, click the Red Flagger extension icon, then choose either `Bio Only` or `Page Audit`.

`Bio Only` evaluates pasted bio text only. `Page Audit` tries to read the page DOM, profile metadata, and profile image evidence automatically.

The extension currently uses TensorFlow.js for exported NLP scoring and JavaScript heuristics for tabular and visual evidence. A real browser-side YOLO model can be added later by exporting a TensorFlow.js graph model to:

```text
chrome_extension/models/yolo_model_tfjs/model.json
```

## Assignment Mapping

| Requirement | Project area |
| --- | --- |
| OkCupid dataset plus synthetic modern dating bios | `data/`, `notebooks/01_data_exploration.ipynb`, `notebooks/02_nlp_model_training.ipynb` |
| Multi-label class imbalance handling | `notebooks/03_tabular_smote_model.ipynb`, `models/tabular_*`, SMOTE tabular module in `api/app.py` |
| Sarcasm and textual tone parsing | TF-IDF NLP model, keyword rules, optional DistilBERT transformer module |
| YOLOv8 visual analysis and IoU | `notebooks/04_yolo_visual_module.ipynb`, `models/yolo_model.pt`, `reports/yolo_visual_metrics.csv` |
| Interactive auditing agent | `api/app.py`, `api/mobile_interface.html`, `notebooks/05_agent_inference_demo.ipynb` |
| Conceptual mobile interface | `http://127.0.0.1:8000/` after starting FastAPI |
| Precision, Recall, and IoU reporting | Evidence Console in the UI and files under `reports/` |
| Chrome extension bonus | `chrome_extension/` |

## Repository Layout

```text
api/                 FastAPI app and mobile-style auditor UI
chrome_extension/    Bonus browser extension prototype
data/                OkCupid-derived dataset and visual sample data
models/              Trained sklearn, DistilBERT, tabular, and YOLO artifacts
notebooks/           Main project workflow, from data prep to final auditor demo
reports/             Generated metrics, JSON assessments, and Markdown reports
scripts/             Utility scripts for export and setup helpers
```

## Notebooks

Run the notebooks in order when regenerating the full pipeline:

```text
01_data_exploration.ipynb
02_nlp_model_training.ipynb
03_tabular_smote_model.ipynb
04_yolo_visual_module.ipynb
05_agent_inference_demo.ipynb
```

## Current Reports

Model metrics and profile audits are written to `reports/`. The UI reads those artifacts so the Evidence Console can show Precision, Recall, YOLO visual subset IoU, structured module outputs, and the final report source.
