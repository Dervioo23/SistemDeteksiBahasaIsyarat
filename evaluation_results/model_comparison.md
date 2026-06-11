# Model Performance Comparison Table
Generated on: 2026-01-11 14:45:49
Data Source: preprocessed_data/alphabet

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| alphabet_baseline_model_final.keras | 0.9385 | 0.9462 | 0.9385 | 0.9391 |
| multiclass_alphabet_model_final.keras | 0.9821 | 0.9837 | 0.9821 | 0.9820 |

## Analysis Notes
- **Baseline Model**: Ideally a simpler architecture (e.g., standard LSTM) without spatial feature extraction (CNN).
- **Proposed Model**: Hybrid CNN-LSTM architecture that extracts both spatial and temporal features.
