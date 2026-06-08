# Dating Profile Audit Report — Profile Typo Red

**Overall result:** High risk (81.85/100).

This is an automated prototype assessment. It should support user decision-making, not judge the person as inherently unsafe.

## 1. Text and bio red flags
- **sarcasm_cynicism** — Sarcastic, dismissive, or cynical tone. Max probability: 0.5050; mean ensemble probability: 0.2687. Sources: tabular, transformer.
- **substance_risk** — Heavy drinking, smoking, drugs, or party-heavy lifestyle. Max probability: 0.9250; mean ensemble probability: 0.4066. Sources: nlp, tabular, transformer.
- **incomplete_profile** — Low-effort profile with very little useful information. Max probability: 0.9541; mean ensemble probability: 0.3842. Sources: nlp, transformer.
- **poor_conflict_resolution** — Cutting people off, refusing communication, revenge, blame, or inability to resolve conflict maturely. Max probability: 0.1271; mean ensemble probability: 0.0424. Sources: nlp.

## 2. Tabular/profile metadata signals
- **sarcasm_cynicism** was triggered by the tabular SMOTE model with probability 0.5050.
- **substance_risk** was triggered by the tabular SMOTE model with probability 0.9250.

## 3. Visual profile image signals
- **Face_Obscured** — The face appears hidden, covered, filtered, or visually unclear.
- Detected persons: 1
- IoU for this row: None
- Visual source: manual_demo_input

## 4. Scoring explanation
- Text risk from strongest labels: 0.7947.
- Visual risk contribution: 0.65.
- Severe interpersonal toxicity labels detected: poor_conflict_resolution.
- Severe-label caution boost: 0.06.

## 5. Model performance note
- NLP model overall metrics: Micro Precision: 0.4780, Micro Recall: 0.7944, Micro F1: 0.5969, Macro Precision: 0.4424, Macro Recall: 0.6586, Macro F1: 0.4988, Hamming Loss: 0.1316.
- Tabular SMOTE model overall metrics: Micro Precision: 0.7308, Micro Recall: 0.3393, Micro F1: 0.4634, Macro Precision: 0.1161, Macro Recall: 0.0776, Macro F1: 0.0926, Hamming Loss: 0.0786.
- YOLO visual subset metrics: Precision 0.75, Recall 0.75, Mean IoU 0.4073.

## 6. Final recommendation
- The app should warn the user to review this profile carefully before engaging.