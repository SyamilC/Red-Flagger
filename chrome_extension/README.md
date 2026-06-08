# Red Flagger Chrome Extension

This is the bonus client-side browser demo. It runs inside Chrome without calling the FastAPI server.

## What Works Now

- Manifest V3 extension.
- Popup UI bundled with esbuild.
- TensorFlow.js bundled into `dist/popup.bundle.js`.
- Client-side TF-IDF logistic regression scoring from `models/nlp_model.json`.
- Browser DOM extraction for selected text or profile-like bio blocks.
- Manual paste box for clean testing when public pages are noisy.
- Client-side keyword, tabular metadata, and visual presentation heuristics.
- No crash when tabular or YOLO TensorFlow.js model folders are missing.

The current extension uses a DOM visual heuristic fallback. Real YOLO TFJS can replace it later if `models/yolo_model_tfjs/model.json` is successfully exported.

## Build

From this folder:

```powershell
npm run check
npm run build
```

The popup loads:

```text
dist/popup.bundle.js
```

## Load In Chrome

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Click Load unpacked.
4. Select:

```text
C:\Users\LENOVO\anaconda3\Red-Flagger\chrome_extension
```

5. Open a profile-like webpage, or any page with a sample dating bio.
6. Click the Red Flagger extension icon.
7. Choose one mode:
   - `Bio Only`: paste one bio into the text box. The extension evaluates only that text.
   - `Page Audit`: leave the text box alone and let the extension find the page bio, nearby metadata, and profile image.
8. Press the audit button.

## After Code Changes

Run:

```powershell
npm run build
```

Then go to `chrome://extensions` and click Reload on the Red Flagger extension.

## Expected Result

The popup should show:

- TFJS ready
- NLP JSON loaded
- Visual DOM heuristic status
- Overall risk score
- Text and metadata flags
- Visual evidence
- Extracted profile summary

If the page is an article full of navigation/header text, do not audit the whole page. Highlight only the dating bio text first, or paste that bio into the popup.

## Audit Modes

`Bio Only` is the cleanest way to test scoring. It ignores page metadata and image checks, so missing profile photos will not affect the result.

`Page Audit` is the bonus extension workflow. It tries to find the best profile-like bio block, then reads metadata and images from the nearest profile card/container.

## Known Limitation

The sklearn tabular Random Forest and PyTorch/Ultralytics YOLO model are not directly browser models. For this bonus extension, the working path is:

- TFJS for exported NLP JSON scoring
- JavaScript heuristics for tabular metadata and visual DOM evidence

To make visual inference truly YOLO-in-browser later, export YOLO to a valid TensorFlow.js graph model and place it at:

```text
models/yolo_model_tfjs/model.json
```
