import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_sample_evaluation_data():
    """Create realistic sample data for BiGRU vs Phi-3 comparison"""

    # Sample URLs with expected labels (based on URL patterns)
    urls = [
        "https://www.google.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "https://www.github.com",
        "https://www.linkedin.com",
        "http://secure-banking-login-verification.suspicious-domain.com",
        "http://paypal-security-update-urgent.fake-domain.net",
        "http://microsoft-account-suspended-click-here.malicious.org",
        "http://apple-id-locked-verify-now.phishing-site.com",
        "http://amazon-security-alert.malicious-site.org",
        "http://facebook-verify-account.phishing-domain.net",
        "http://banking-urgent-action.scam-site.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.netflix.com",
        "http://netflix-billing-problem.fake-domain.org",
        "http://google-account-suspended.phishing.net",
        "http://linkedin-security-notice.malicious.com",
        "https://www.twitter.com",
        "http://twitter-verify-blue.scam-domain.net"
    ]

    # Ground truth labels (0 = legitimate, 1 = phishing)
    true_labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]

    # Simulate BiGRU predictions (some false positives and false negatives)
    np.random.seed(42)
    bigru_predictions = []
    bigru_confidences = []

    for i, true_label in enumerate(true_labels):
        if true_label == 0:  # Legitimate URLs
            # BiGRU sometimes misclassifies legitimate URLs as phishing (false positives)
            if np.random.random() < 0.15:  # 15% false positive rate
                pred = 1
                conf = np.random.uniform(0.55, 0.75)
            else:
                pred = 0
                conf = np.random.uniform(0.15, 0.45)
        else:  # Phishing URLs
            # BiGRU sometimes misses phishing URLs (false negatives)
            if np.random.random() < 0.10:  # 10% false negative rate
                pred = 0
                conf = np.random.uniform(0.25, 0.48)
            else:
                pred = 1
                conf = np.random.uniform(0.55, 0.85)

        bigru_predictions.append(pred)
        bigru_confidences.append(conf)

    # Simulate Phi-3 predictions (improved accuracy, fewer false positives)
    phi3_predictions = []
    used_layer2 = []

    for i, (true_label, bigru_conf) in enumerate(zip(true_labels, bigru_confidences)):
        # URLs with confidence between 0.3-0.7 go to Layer 2 (Phi-3)
        if 0.3 < bigru_conf < 0.7:
            used_layer2.append(True)
            # Phi-3 is more accurate, especially at reducing false positives
            if true_label == 0:  # Legitimate URLs
                if np.random.random() < 0.05:  # Only 5% false positive rate
                    pred = 1
                else:
                    pred = 0
            else:  # Phishing URLs
                if np.random.random() < 0.03:  # Only 3% false negative rate
                    pred = 0
                else:
                    pred = 1
            phi3_predictions.append(pred)
        else:
            used_layer2.append(False)
            phi3_predictions.append(bigru_predictions[i])

    # Create DataFrame
    data = {
        'url': urls,
        'true_label': true_labels,
        'layer1_confidence': bigru_confidences,
        'layer1_prediction': bigru_predictions,
        'used_layer2': used_layer2,
        'final_prediction': phi3_predictions,
        'decision_maker': ['Phi3' if used else 'BiGRU' for used in used_layer2]
    }

    return pd.DataFrame(data)

def plot_performance_comparison(df):
    """Create bar chart comparing F1-scores and Accuracy between BiGRU and Phi-3"""

    y_true = df['true_label'].values
    y_pred_bigru = df['layer1_prediction'].values
    y_pred_hybrid = df['final_prediction'].values

    # Calculate metrics
    metrics_bigru = {
        'Accuracy': accuracy_score(y_true, y_pred_bigru),
        'Precision': precision_score(y_true, y_pred_bigru, zero_division=0),
        'Recall': recall_score(y_true, y_pred_bigru, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_bigru, zero_division=0)
    }

    metrics_hybrid = {
        'Accuracy': accuracy_score(y_true, y_pred_hybrid),
        'Precision': precision_score(y_true, y_pred_hybrid, zero_division=0),
        'Recall': recall_score(y_true, y_pred_hybrid, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_hybrid, zero_division=0)
    }

    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    models = ['BiGRU Only', 'BiGRU + Phi-3\n(Hybrid)']
    colors = ['#3498db', '#e74c3c']

    # Accuracy comparison
    accuracy_values = [metrics_bigru['Accuracy'], metrics_hybrid['Accuracy']]
    bars1 = ax1.bar(models, accuracy_values, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    for i, v in enumerate(accuracy_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Precision comparison
    precision_values = [metrics_bigru['Precision'], metrics_hybrid['Precision']]
    bars2 = ax2.bar(models, precision_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Model Precision Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    for i, v in enumerate(precision_values):
        ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Recall comparison
    recall_values = [metrics_bigru['Recall'], metrics_hybrid['Recall']]
    bars3 = ax3.bar(models, recall_values, color=colors, alpha=0.8)
    ax3.set_ylabel('Recall', fontsize=12)
    ax3.set_title('Model Recall Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 1)
    for i, v in enumerate(recall_values):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # F1-Score comparison
    f1_values = [metrics_bigru['F1-Score'], metrics_hybrid['F1-Score']]
    bars4 = ax4.bar(models, f1_values, color=colors, alpha=0.8)
    ax4.set_ylabel('F1-Score', fontsize=12)
    ax4.set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1)
    for i, v in enumerate(f1_values):
        ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print improvement summary
    print("PERFORMANCE IMPROVEMENT SUMMARY")
    print("=" * 40)
    print(f"Accuracy improvement: {metrics_hybrid['Accuracy'] - metrics_bigru['Accuracy']:+.3f}")
    print(f"Precision improvement: {metrics_hybrid['Precision'] - metrics_bigru['Precision']:+.3f}")
    print(f"Recall improvement: {metrics_hybrid['Recall'] - metrics_bigru['Recall']:+.3f}")
    print(f"F1-Score improvement: {metrics_hybrid['F1-Score'] - metrics_bigru['F1-Score']:+.3f}")

    return metrics_bigru, metrics_hybrid

def plot_confusion_matrices(df):
    """Create confusion matrices showing false positive reduction"""

    y_true = df['true_label'].values
    y_pred_bigru = df['layer1_prediction'].values
    y_pred_hybrid = df['final_prediction'].values

    # Calculate confusion matrices
    cm_bigru = confusion_matrix(y_true, y_pred_bigru)
    cm_hybrid = confusion_matrix(y_true, y_pred_hybrid)

    # Create side-by-side confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # BiGRU confusion matrix
    sns.heatmap(cm_bigru, annot=True, fmt='d', cmap='Blues', ax=ax1,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_title('BiGRU Only\nConfusion Matrix', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xticklabels(['Legitimate', 'Phishing'], fontsize=11)
    ax1.set_yticklabels(['Legitimate', 'Phishing'], fontsize=11, rotation=0)

    # Add false positive highlighting
    if cm_bigru.shape == (2, 2):
        ax1.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False, edgecolor='red', lw=3))
        ax1.text(1.5, 0.5, f'FP: {cm_bigru[0, 1]}', ha='center', va='center',
                color='red', fontsize=12, fontweight='bold')

    # Hybrid confusion matrix
    sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', ax=ax2,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_title('BiGRU + Phi-3 (Hybrid)\nConfusion Matrix', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xticklabels(['Legitimate', 'Phishing'], fontsize=11)
    ax2.set_yticklabels(['Legitimate', 'Phishing'], fontsize=11, rotation=0)

    # Add false positive highlighting
    if cm_hybrid.shape == (2, 2):
        ax2.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False, edgecolor='red', lw=3))
        ax2.text(1.5, 0.5, f'FP: {cm_hybrid[0, 1]}', ha='center', va='center',
                color='red', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Calculate and display false positive reduction
    if cm_bigru.shape == (2, 2) and cm_hybrid.shape == (2, 2):
        fp_bigru = cm_bigru[0, 1]  # False positives (legitimate classified as phishing)
        fp_hybrid = cm_hybrid[0, 1]
        fp_reduction = fp_bigru - fp_hybrid
        fp_reduction_percent = (fp_reduction / max(fp_bigru, 1)) * 100

        print("\nFALSE POSITIVE ANALYSIS")
        print("=" * 30)
        print(f"BiGRU False Positives: {fp_bigru}")
        print(f"Hybrid False Positives: {fp_hybrid}")
        print(f"False Positive Reduction: {fp_reduction} ({fp_reduction_percent:.1f}%)")

        # Also show false negatives
        fn_bigru = cm_bigru[1, 0]  # False negatives (phishing classified as legitimate)
        fn_hybrid = cm_hybrid[1, 0]
        fn_reduction = fn_bigru - fn_hybrid
        fn_reduction_percent = (fn_reduction / max(fn_bigru, 1)) * 100

        print(f"\nFalse Negative Analysis:")
        print(f"BiGRU False Negatives: {fn_bigru}")
        print(f"Hybrid False Negatives: {fn_hybrid}")
        print(f"False Negative Reduction: {fn_reduction} ({fn_reduction_percent:.1f}%)")

    return cm_bigru, cm_hybrid

def create_detailed_analysis_report(df, metrics_bigru, metrics_hybrid):
    """Create a detailed analysis report"""

    print("\nDETAILED ANALYSIS REPORT")
    print("=" * 50)

    total_samples = len(df)
    layer2_samples = df['used_layer2'].sum()
    layer2_percentage = (layer2_samples / total_samples) * 100

    print(f"\nDataset Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Legitimate URLs: {(df['true_label'] == 0).sum()}")
    print(f"Phishing URLs: {(df['true_label'] == 1).sum()}")
    print(f"Processed by Layer 2 (Phi-3): {layer2_samples} ({layer2_percentage:.1f}%)")

    print(f"\nModel Performance Comparison:")
    print("-" * 30)
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        bigru_val = metrics_bigru[metric]
        hybrid_val = metrics_hybrid[metric]
        improvement = hybrid_val - bigru_val
        print(f"{metric:12}: BiGRU={bigru_val:.3f}, Hybrid={hybrid_val:.3f}, Improvement={improvement:+.3f}")

    # Analyze Layer 2 impact
    layer2_df = df[df['used_layer2'] == True]
    if len(layer2_df) > 0:
        print(f"\nLayer 2 (Phi-3) Impact Analysis:")
        print("-" * 30)
        corrections = layer2_df[
            (layer2_df['layer1_prediction'] != layer2_df['final_prediction']) &
            (layer2_df['final_prediction'] == layer2_df['true_label'])
        ]
        mistakes = layer2_df[
            (layer2_df['layer1_prediction'] != layer2_df['final_prediction']) &
            (layer2_df['final_prediction'] != layer2_df['true_label'])
        ]

        print(f"Correct corrections by Phi-3: {len(corrections)}")
        print(f"Incorrect changes by Phi-3: {len(mistakes)}")
        print(f"Net improvement rate: {(len(corrections) - len(mistakes)) / len(layer2_df) * 100:.1f}%")

def main():
    """Main function to create all visualizations"""

    print("BiGRU vs Phi-3 Performance Comparison")
    print("=" * 50)

    # Create sample data
    print("Creating evaluation dataset...")
    df = create_sample_evaluation_data()

    # Create performance comparison bar charts
    print("\nGenerating performance comparison charts...")
    metrics_bigru, metrics_hybrid = plot_performance_comparison(df)

    # Create confusion matrices
    print("\nGenerating confusion matrices...")
    cm_bigru, cm_hybrid = plot_confusion_matrices(df)

    # Create detailed analysis
    create_detailed_analysis_report(df, metrics_bigru, metrics_hybrid)

    # Save the evaluation data
    df.to_csv('model_comparison_data.csv', index=False)
    print(f"\nEvaluation data saved to: model_comparison_data.csv")
    print("Charts saved as: model_performance_comparison.png, confusion_matrices_comparison.png")

if __name__ == "__main__":
    main()
