# Chrome Extension Model Assets

This directory is the expected browser-side model location.

Expected files:

```text
models/nlp_model.json
models/tabular_model_tfjs/model.json
models/yolo_model_tfjs/model.json
```

Generate the NLP JSON starter with:

```bash
python scripts/export_nlp_for_tfjs.py
```

Generate YOLO TensorFlow.js assets with:

```bash
python scripts/export_yolo_for_tfjs.py
```

The tabular SMOTE Random Forest artifact is not directly compatible with `tf.loadLayersModel`. Use a JSON scorer or train/export a small neural surrogate if the extension must keep tabular inference fully client-side.
