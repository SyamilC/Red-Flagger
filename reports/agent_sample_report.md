# Dating Profile Audit Report — Profile 0

**Overall result:** Moderate risk (45.06/100).

This is an automated prototype assessment. It should support user decision-making, not judge the person as inherently unsafe.

## 1. Text and bio red flags
- **sarcasm_cynicism** — Sarcastic, dismissive, or cynical tone. Max probability: 0.9191; mean ensemble probability: 0.5277. Sources: nlp, tabular.

## 2. Tabular/profile metadata signals
- **sarcasm_cynicism** was triggered by the tabular SMOTE model with probability 0.8000.

## 3. Visual profile image signals
- **clean** — No major visual presentation issue detected.
- Detected persons: 1
- IoU for this row: 0.4372
- Visual source: yolo_live

## 4. Scoring explanation
- Text risk from strongest labels: 0.6008.
- Visual risk contribution: 0.0.
- Severe interpersonal toxicity labels detected: None.
- Severe-label caution boost: 0.0.

## 5. Model performance note
- NLP model overall metrics: Micro Precision: 0.4780, Micro Recall: 0.7944, Micro F1: 0.5969, Macro Precision: 0.4424, Macro Recall: 0.6586, Macro F1: 0.4988, Hamming Loss: 0.1316.
- Tabular SMOTE model overall metrics: Micro Precision: 0.7308, Micro Recall: 0.3393, Micro F1: 0.4634, Macro Precision: 0.1161, Macro Recall: 0.0776, Macro F1: 0.0926, Hamming Loss: 0.0786.
- YOLO visual subset metrics: Precision 0.75, Recall 0.75, Mean IoU 0.4073.

## 6. Final recommendation
- The app should show a caution note and highlight the specific evidence.