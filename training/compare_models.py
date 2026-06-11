import os
import argparse
import numpy as np
import pickle
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

def load_data(data_dir):
    """Muat data uji dari direktori preprocessing"""
    print(f"📂 Loading test data from: {data_dir}")
    X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return X_test, y_test.flatten(), metadata

def evaluate_model(model_path, X_test, y_true):
    """Evaluasi satu model dan kembalikan metriknya"""
    model_basename = os.path.basename(model_path)
    print(f"📦 Evaluating model: {model_basename}")
    
    model = keras.models.load_model(model_path)
    y_pred_probs = model.predict(X_test, verbose=0)
    
    if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] > 1:
        y_pred = np.argmax(y_pred_probs, axis=1)
        average_mode = 'macro'
    else:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        average_mode = 'binary'
        
    metrics = {
        'Model': model_basename,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average=average_mode, zero_division=0),
        'Recall': recall_score(y_true, y_pred, average=average_mode, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, average=average_mode, zero_division=0)
    }
    return metrics

def generate_markdown_table(results):
    """Hasilkan string tabel markdown dari hasil evaluasi"""
    header = "| Model | Accuracy | Precision | Recall | F1-Score |"
    separator = "| :--- | :---: | :---: | :---: | :---: |"
    rows = []
    
    for r in results:
        row = f"| {r['Model']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1-Score']:.4f} |"
        rows.append(row)
        
    return "\n".join([header, separator] + rows)

def main():
    parser = argparse.ArgumentParser(description='Compare multiple models and generate a performance table')
    parser.add_argument('--models', nargs='+', required=True, help='Paths to .keras model files')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing preprocessed test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save the comparison report')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    X_test, y_true, metadata = load_data(args.data_dir)
    
    # 2. Evaluate each model
    results = []
    for model_path in args.models:
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model not found at {model_path}")
            continue
        metrics = evaluate_model(model_path, X_test, y_true)
        results.append(metrics)
    
    if not results:
        print("❌ No models were successfully evaluated.")
        return

    # 3. Generate Report
    table_md = generate_markdown_table(results)
    
    report_content = f"""# Model Performance Comparison Table
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Data Source: {args.data_dir}

{table_md}

## Analysis Notes
- **Baseline Model**: Ideally a simpler architecture (e.g., standard LSTM) without spatial feature extraction (CNN).
- **Proposed Model**: Hybrid CNN-LSTM architecture that extracts both spatial and temporal features.
"""

    report_path = os.path.join(args.output_dir, 'model_comparison.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
        
    print(f"\n✅ Comparison report generated: {report_path}")
    print("\n" + table_md)

if __name__ == '__main__':
    main()
