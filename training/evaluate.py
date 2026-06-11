import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

def load_model_and_data(
    model_path: str,
    data_dir: str
):
    """
    Muat model yang telah dilatih dan data uji
    
    Args:
        model_path: Path ke model yang disimpan (.keras)
        data_dir: Direktori dengan data yang telah diproses
        
    Returns:
        Tuple dari (model, X_test, y_test, metadata)
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL AND DATA")
    print(f"{'='*60}\n")
    
    # Muat model
    print(f"📦 Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"   ✅ Model loaded")
    
    # Muat data uji
    print(f"\n📂 Loading test data from: {data_dir}")
    X_test = np.load(os.path.join(data_dir, 'test_X.npy'))
    y_test = np.load(os.path.join(data_dir, 'test_y.npy'))
    
    with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"   ✅ Data loaded")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Input shape: {X_test.shape}")
    
    return model, X_test, y_test, metadata


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list = None
) -> dict:
    """
    Evaluasi model yang komprehensif
    
    Args:
        model: Model Keras yang telah dilatih
        X_test: Fitur uji
        y_test: Label uji
        class_names: Nama kelas
        
    Returns:
        Dictionary dengan metrik evaluasi
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}\n")
    
    # Prediksi
    print("🔮 Making predictions...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_true = y_test.flatten()

    # Tangani output biner vs multi-kelas
    if y_pred_probs.ndim == 2 and y_pred_probs.shape[1] > 1:
        # Multi-kelas: probabilitas per kelas
        y_pred = np.argmax(y_pred_probs, axis=1)
        average_mode = 'macro'
    else:
        # Biner: output probabilitas tunggal
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
        average_mode = 'binary'
    
    # Metrik dasar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)[:2]
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Metrik terperinci
    accuracy = accuracy_score(y_true, y_pred)
    if average_mode == 'binary':
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n📈 Detailed Metrics:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Laporan klasifikasi
    if class_names is None:
        unique_labels = np.unique(y_true)
        class_names = [f'Class {int(i)}' for i in unique_labels]
    
    print(f"\n📊 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"📊 Confusion Matrix:")
    print(cm)
    
    # Kumpulkan metrik
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_probs': y_pred_probs,
        'classification_report': report,
        'average_mode': average_mode,
        'class_names': class_names,
    }
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    save_path: str = None
):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: str = None
):
    """Plot kurva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")
    
    plt.show()
    
    return roc_auc


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: list,
    save_path: str = None
):
    """Plot distribusi probabilitas prediksi"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Prediksi Kelas 0
    class_0_probs = y_pred_probs[y_true == 0]
    axes[0].hist(class_0_probs, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    axes[0].set_title(f'Prediction Distribution for {class_names[0]}')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Prediksi Kelas 1
    class_1_probs = y_pred_probs[y_true == 1]
    axes[1].hist(class_1_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    axes[1].set_title(f'Prediction Distribution for {class_names[1]}')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Count')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution plot saved: {save_path}")
    
    plt.show()


def save_evaluation_report(
    metrics: dict,
    model_name: str,
    output_dir: str = 'evaluation_results'
):
    """Simpan laporan evaluasi ke file"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, f'{model_name}_evaluation.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"Model Evaluation Report: {model_name}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Test Loss: {metrics['test_loss']:.4f}\n")
        f.write(f"  Test Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics['precision']:.4f}\n")
        f.write(f"  Recall: {metrics['recall']:.4f}\n")
        f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n")
        f.write("\nClassification Report:\n")
        f.write(metrics.get('classification_report', '') + "\n")
    
    print(f"✅ Evaluation report saved: {report_path}")


def main():
    """Fungsi evaluasi utama"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION TOOL")
    print("="*60)
    
    # Konfigurasi CLI
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model_path', type=str, default='trained_models/word_halo_model_best.keras',
                        help='Path to trained model (.keras)')
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/words',
                        help='Directory containing preprocessed data (with test_X.npy, test_y.npy, metadata.pkl)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, default='word_halo_model',
                        help='Name used in report/plot filenames')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plotting (only print metrics and save report)')
    args = parser.parse_args()

    model_path = args.model_path
    data_dir = args.data_dir
    output_dir = args.output_dir
    model_name = args.model_name
    
    # Periksa apakah model ada
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print(f"   Please train the model first!")
        return
    
    # Muat
    model, X_test, y_test, metadata = load_model_and_data(model_path, data_dir)
    
    # Evaluasi
    class_names = metadata.get('class_names')
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # Visualisasi
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)

    # Gunakan nama kelas dari metrik untuk memastikan konsistensi
    used_class_names = metrics.get('class_names', class_names)

    # Confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        used_class_names,
        save_path=os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    )

    # Hanya hasilkan plot ROC / distribusi untuk model biner
    if not args.no_plots:
        y_pred_probs = metrics['y_pred_probs']
        is_binary_probs = (
            y_pred_probs.ndim == 1 or
            (y_pred_probs.ndim == 2 and y_pred_probs.shape[1] == 1)
        )
        if is_binary_probs:
            # Untuk bentuk 2D (N,1), ratakan ke 1D untuk metrik
            if y_pred_probs.ndim == 2:
                y_probs_1d = y_pred_probs.flatten()
            else:
                y_probs_1d = y_pred_probs

            roc_auc = plot_roc_curve(
                metrics['y_true'],
                y_probs_1d,
                save_path=os.path.join(output_dir, f'{model_name}_roc_curve.png')
            )
            print(f"\n📊 ROC AUC Score: {roc_auc:.4f}")

            plot_prediction_distribution(
                metrics['y_true'],
                y_probs_1d,
                used_class_names,
                save_path=os.path.join(output_dir, f'{model_name}_distribution.png')
            )
        else:
            print("\nℹ️  Skipping ROC and probability distribution plots for multi-class model.")

    # Simpan laporan
    save_evaluation_report(metrics, model_name, output_dir)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED!")
    print(f"{'='*60}")
    print(f"\nResults saved in: {output_dir}/")


if __name__ == '__main__':
    main()
