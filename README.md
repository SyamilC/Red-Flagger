# Red-Flagger

Dating Red Flag Detector is an interactive profile auditing prototype for multi-label dating profile risk signals. The main demo is now the local FastAPI mobile-style UI, which combines bio text, OkCupid-style metadata, optional profile photo upload, YOLO visual analysis, and an explainable audit report.

The Chrome extension remains a bonus/browser-side experiment. For grading and demonstration, start with the local UI.

## What The App Does

- Audits dating bios for sarcasm, cynicism, controlling language, negativity, entitlement, hookup focus, substance risk, and related labels.
- Uses OkCupid-style profile metadata such as age, body type, education, children/offspring, smoking, drinking, drugs, job, pets, religion, sign, and more.
- Handles imbalanced tabular classification through the SMOTE-trained tabular model.
- Accepts an optional profile photo and runs YOLOv8 automatically as part of the same full audit.
- Shows Precision, Recall, and visual IoU metrics from the generated reports.
- Produces an end-user audit report using either the local fallback writer or OpenRouter when a key is provided.

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

## Environment Setup

From PowerShell:

```powershell
cd C:\Users\LENOVO\anaconda3\Red-Flagger
```

If `conda` works in your terminal:

```powershell
conda env create -f environment.yml
conda activate red-flagger
```

If `conda` is not recognized, use the existing environment Python directly:

```powershell
& "C:\Users\LENOVO\anaconda3\envs\red-flagger\python.exe" -m pip install -r requirements.txt
```

Useful dependency checks:

```powershell
& "C:\Users\LENOVO\anaconda3\envs\red-flagger\python.exe" -c "import fastapi, multipart, ultralytics; print('api deps ok')"
```

The optional transformer module needs Hugging Face Transformers:

```powershell
& "C:\Users\LENOVO\anaconda3\envs\red-flagger\python.exe" -m pip install transformers accelerate safetensors
```

## Run The Mobile Auditor UI

Start the local API:

```powershell
cd C:\Users\LENOVO\anaconda3\Red-Flagger
& "C:\Users\LENOVO\anaconda3\envs\red-flagger\python.exe" -m uvicorn api.app:app --reload
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

## Report Source: `local_fallback` vs `openrouter`

The models still run either way. This badge only describes who wrote the final prose report.

- `local_fallback`: the app used the built-in local report writer because OpenRouter was not requested, no key was available, or the OpenRouter request failed.
- `openrouter`: the app successfully used OpenRouter to beautify the final report.

You can provide an OpenRouter key in either place:

- UI: paste it into `Beautify key` before running an audit.
- Environment variable:

```powershell
$env:OPENROUTER_API_KEY="your_key_here"
```

Then run the audit with beautification enabled.

## API Checks

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Sample audit:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/audit/sample/0
```

Photo upload audit:

```powershell
curl.exe -X POST http://127.0.0.1:8000/audit/photo `
  -F "bio=Fluent in sarcasm and always right." `
  -F "profile_id=demo_photo" `
  -F "tabular_json={}" `
  -F "use_openrouter=false" `
  -F "photo=@C:\Users\LENOVO\anaconda3\Red-Flagger\data\test_images\profile_000.jpg;type=image/jpeg"
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

## Chrome Extension Bonus

The Chrome extension is not the primary demo path. It is kept as the extra client-side/browser-side component under `chrome_extension/`.

Basic checks:

```powershell
node --check chrome_extension/content.js
node --check chrome_extension/popup.js
node --check chrome_extension/inference.js
```

Manual loading:

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked and select `chrome_extension/`.

## Current Reports

Model metrics and profile audits are written to `reports/`. The UI reads those artifacts so the Evidence Console can show Precision, Recall, YOLO visual subset IoU, structured module outputs, and the final report source.
