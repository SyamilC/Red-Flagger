# Dating Profile Audit Report — Profile Absolute Red

**Overall result:** High risk (87.36/100).

This is an automated prototype assessment. It should support user decision-making, not judge the person as inherently unsafe.

## 1. Text and bio red flags
- **substance_risk** — Heavy drinking, smoking, drugs, or party-heavy lifestyle. Max probability: 0.7950; mean ensemble probability: 0.3258. Sources: tabular, transformer.
- **emotional_manipulation** — Silent treatment, guilt, emotional punishment, testing, or conditional affection. Max probability: 0.9996; mean ensemble probability: 0.3332. Sources: nlp.
- **entitlement_superiority** — Self-important, superior, dismissive, or 'people must impress me' framing. Max probability: 0.6483; mean ensemble probability: 0.2161. Sources: nlp.
- **poor_conflict_resolution** — Cutting people off, refusing communication, revenge, blame, or inability to resolve conflict maturely. Max probability: 0.9799; mean ensemble probability: 0.3266. Sources: nlp.

## 2. Tabular/profile metadata signals
- **substance_risk** was triggered by the tabular SMOTE model with probability 0.7950.

## 3. Visual profile image signals
- **clean** — No major visual presentation issue detected.
- Detected persons: 1
- IoU for this row: None
- Visual source: manual_demo_input

## 4. Scoring explanation
- Text risk from strongest labels: 0.9248.
- Visual risk contribution: 0.0.
- Severe interpersonal toxicity labels detected: emotional_manipulation, entitlement_superiority, poor_conflict_resolution.
- Severe-label caution boost: 0.18.

## 5. Model performance note
- NLP model overall metrics: Micro Precision: 0.4780, Micro Recall: 0.7944, Micro F1: 0.5969, Macro Precision: 0.4424, Macro Recall: 0.6586, Macro F1: 0.4988, Hamming Loss: 0.1316.
- Tabular SMOTE model overall metrics: Micro Precision: 0.7308, Micro Recall: 0.3393, Micro F1: 0.4634, Macro Precision: 0.1161, Macro Recall: 0.0776, Macro F1: 0.0926, Hamming Loss: 0.0786.
- YOLO visual subset metrics: Precision 0.75, Recall 0.75, Mean IoU 0.4073.

## 6. Final recommendation
- The app should warn the user to review this profile carefully before engaging.