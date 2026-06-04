# Dating Profile Audit Report — Profile 0

**Overall result:** Low-to-Moderate risk (26.95/100).

This is an automated prototype assessment. It should support user decision-making, not judge the person as inherently unsafe.

## 1. Text and bio red flags
- **sarcasm_cynicism** — Sarcastic or dismissive tone that may need careful interpretation. Mean ensemble probability: 0.5980.

## 2. Tabular/profile metadata signals
- **sarcasm_cynicism** was triggered by the tabular SMOTE model with probability 0.7700.

## 3. Visual profile image signals
- **clean** — No major visual presentation issue detected.
- Detected persons: 1
- IoU for this row: 0.4372
- Visual source: yolo_live

## 4. Model performance note
- NLP model overall metrics: Micro Precision: 0.4898, Micro Recall: 0.4103, Micro F1: 0.4465, Macro Precision: 0.3468, Macro Recall: 0.3268, Macro F1: 0.3092, Hamming Loss: 0.1417.
- Tabular SMOTE model overall metrics: Micro Precision: 0.8600, Micro Recall: 0.3675, Micro F1: 0.5150, Macro Precision: 0.1593, Macro Recall: 0.1156, Macro F1: 0.1340, Hamming Loss: 0.0964.
- YOLO visual subset metrics: Precision 0.75, Recall 0.75, Mean IoU 0.4073.

## 5. Final recommendation
- The app can mark the profile as relatively low-risk while still showing transparent model evidence.