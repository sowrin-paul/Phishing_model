import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import json
import os

class HybridModelEvaluator:
    """Comprehensive evaluation tool for the hybrid phishing detection system"""

    def __init__(self, results_file: str = None, results_df: pd.DataFrame = None):
        if results_df is not None:
            self.df = results_df
        elif results_file:
            self.df = pd.read_csv(results_file)
        else:
            raise ValueError("Either results_file or results_df must be provided")

        self.metrics = {}

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        if 'true_label' not in self.df.columns:
            print("No ground truth labels found. Cannot calculate evaluation metrics.")
            return {}

        y_true = self.df['true_label'].values
        y_pred = self.df['final_prediction'].values
        y_pred_layer1 = self.df['layer1_prediction'].values

        # Overall metrics
        self.metrics['overall'] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
        }

        # Layer 1 (BiGRU) only metrics
        self.metrics['layer1_only'] = {
            'accuracy': accuracy_score(y_true, y_pred_layer1),
            'precision': precision_score(y_true, y_pred_layer1, zero_division=0),
            'recall': recall_score(y_true, y_pred_layer1, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_layer1, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred_layer1) if len(np.unique(y_true)) > 1 else 0
        }

        # Layer 2 impact analysis
        layer2_mask = self.df['used_layer2'] == True
        if layer2_mask.sum() > 0:
            y_true_layer2 = y_true[layer2_mask]
            y_pred_layer2 = y_pred[layer2_mask]
            y_pred_layer1_layer2 = y_pred_layer1[layer2_mask]

            self.metrics['layer2_samples'] = {
                'count': int(layer2_mask.sum()),
                'accuracy_improvement': accuracy_score(y_true_layer2, y_pred_layer2) - accuracy_score(y_true_layer2, y_pred_layer1_layer2),
                'precision_layer2': precision_score(y_true_layer2, y_pred_layer2, zero_division=0),
                'recall_layer2': recall_score(y_true_layer2, y_pred_layer2, zero_division=0),
                'f1_layer2': f1_score(y_true_layer2, y_pred_layer2, zero_division=0)
            }

        # Efficiency metrics
        total_samples = len(self.df)
        layer1_resolved = total_samples - layer2_mask.sum()

        self.metrics['efficiency'] = {
            'total_samples': total_samples,
            'layer1_resolved': int(layer1_resolved),
            'layer2_processed': int(layer2_mask.sum()),
            'efficiency_percentage': (layer1_resolved / total_samples) * 100,
            'layer2_usage_percentage': (layer2_mask.sum() / total_samples) * 100
        }

        return self.metrics

    def print_detailed_report(self):
        """Print a comprehensive evaluation report"""
        if not self.metrics:
            self.calculate_metrics()

        print("HYBRID PHISHING DETECTION - EVALUATION REPORT")
        print("=" * 60)

        # Overall Performance
        print("\n📊 OVERALL PERFORMANCE")
        print("-" * 30)
        overall = self.metrics.get('overall', {})
        print(f"Accuracy:  {overall.get('accuracy', 0):.4f}")
        print(f"Precision: {overall.get('precision', 0):.4f}")
        print(f"Recall:    {overall.get('recall', 0):.4f}")
        print(f"F1-Score:  {overall.get('f1_score', 0):.4f}")
        print(f"AUC-ROC:   {overall.get('auc_roc', 0):.4f}")

        # Layer 1 Performance
        print("\nLAYER 1 (BiGRU) ONLY PERFORMANCE")
        print("-" * 35)
        layer1 = self.metrics.get('layer1_only', {})
        print(f"Accuracy:  {layer1.get('accuracy', 0):.4f}")
        print(f"Precision: {layer1.get('precision', 0):.4f}")
        print(f"Recall:    {layer1.get('recall', 0):.4f}")
        print(f"F1-Score:  {layer1.get('f1_score', 0):.4f}")
        print(f"AUC-ROC:   {layer1.get('auc_roc', 0):.4f}")

        # Layer 2 Impact
        if 'layer2_samples' in self.metrics:
            print("\nLAYER 2 (Phi3) IMPACT ANALYSIS")
            print("-" * 35)
            layer2 = self.metrics['layer2_samples']
            print(f"Samples processed by Layer 2: {layer2['count']}")
            print(f"Accuracy improvement: {layer2['accuracy_improvement']:+.4f}")
            print(f"Layer 2 Precision: {layer2['precision_layer2']:.4f}")
            print(f"Layer 2 Recall:    {layer2['recall_layer2']:.4f}")
            print(f"Layer 2 F1-Score:  {layer2['f1_layer2']:.4f}")

        # Efficiency Analysis
        print("\nEFFICIENCY ANALYSIS")
        print("-" * 25)
        efficiency = self.metrics.get('efficiency', {})
        print(f"Total samples: {efficiency.get('total_samples', 0)}")
        print(f"Resolved by Layer 1: {efficiency.get('layer1_resolved', 0)}")
        print(f"Processed by Layer 2: {efficiency.get('layer2_processed', 0)}")
        print(f"Pipeline efficiency: {efficiency.get('efficiency_percentage', 0):.1f}%")
        print(f"Layer 2 usage: {efficiency.get('layer2_usage_percentage', 0):.1f}%")

        # Performance Comparison
        if 'overall' in self.metrics and 'layer1_only' in self.metrics:
            print("\nHYBRID vs LAYER 1 ONLY COMPARISON")
            print("-" * 40)
            acc_improvement = self.metrics['overall']['accuracy'] - self.metrics['layer1_only']['accuracy']
            f1_improvement = self.metrics['overall']['f1_score'] - self.metrics['layer1_only']['f1_score']
            print(f"Accuracy improvement: {acc_improvement:+.4f}")
            print(f"F1-Score improvement: {f1_improvement:+.4f}")

    def plot_performance_comparison(self, save_path: str = None):
        """Create visualization comparing different model configurations"""
        if not self.metrics:
            self.calculate_metrics()

        # Prepare data for plotting
        models = ['Layer 1 Only\n(BiGRU)', 'Hybrid System\n(BiGRU + Phi3)']

        if 'layer1_only' not in self.metrics or 'overall' not in self.metrics:
            print("Insufficient data for performance comparison plot")
            return

        accuracy = [self.metrics['layer1_only']['accuracy'], self.metrics['overall']['accuracy']]
        precision = [self.metrics['layer1_only']['precision'], self.metrics['overall']['precision']]
        recall = [self.metrics['layer1_only']['recall'], self.metrics['overall']['recall']]
        f1_score = [self.metrics['layer1_only']['f1_score'], self.metrics['overall']['f1_score']]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Accuracy
        bars1 = ax1.bar(models, accuracy, color=['#3498db', '#e74c3c'])
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracy):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        # Precision
        bars2 = ax2.bar(models, precision, color=['#3498db', '#e74c3c'])
        ax2.set_ylabel('Precision')
        ax2.set_title('Model Precision Comparison')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(precision):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        # Recall
        bars3 = ax3.bar(models, recall, color=['#3498db', '#e74c3c'])
        ax3.set_ylabel('Recall')
        ax3.set_title('Model Recall Comparison')
        ax3.set_ylim(0, 1)
        for i, v in enumerate(recall):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        # F1-Score
        bars4 = ax4.bar(models, f1_score, color=['#3498db', '#e74c3c'])
        ax4.set_ylabel('F1-Score')
        ax4.set_title('Model F1-Score Comparison')
        ax4.set_ylim(0, 1)
        for i, v in enumerate(f1_score):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance comparison plot saved to: {save_path}")

        plt.show()

    def plot_confusion_matrices(self, save_path: str = None):
        """Plot confusion matrices for both hybrid and layer 1 only"""
        if 'true_label' not in self.df.columns:
            print("No ground truth labels for confusion matrix")
            return

        y_true = self.df['true_label'].values
        y_pred_hybrid = self.df['final_prediction'].values
        y_pred_layer1 = self.df['layer1_prediction'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Layer 1 only confusion matrix
        cm1 = confusion_matrix(y_true, y_pred_layer1)
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Layer 1 Only (BiGRU)\nConfusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.set_xticklabels(['Legitimate', 'Phishing'])
        ax1.set_yticklabels(['Legitimate', 'Phishing'])

        # Hybrid system confusion matrix
        cm2 = confusion_matrix(y_true, y_pred_hybrid)
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', ax=ax2)
        ax2.set_title('Hybrid System (BiGRU + Phi3)\nConfusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_xticklabels(['Legitimate', 'Phishing'])
        ax2.set_yticklabels(['Legitimate', 'Phishing'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")

        plt.show()

    def analyze_layer2_effectiveness(self):
        """Analyze the effectiveness of Layer 2 (Phi3) intervention"""
        if 'used_layer2' not in self.df.columns or 'true_label' not in self.df.columns:
            print("Cannot analyze Layer 2 effectiveness - missing required columns")
            return

        layer2_samples = self.df[self.df['used_layer2'] == True]

        if len(layer2_samples) == 0:
            print("No samples were processed by Layer 2")
            return

        print("\nLAYER 2 EFFECTIVENESS ANALYSIS")
        print("=" * 40)

        # Calculate corrections made by Layer 2
        corrections = layer2_samples[
            (layer2_samples['layer1_prediction'] != layer2_samples['final_prediction']) &
            (layer2_samples['final_prediction'] == layer2_samples['true_label'])
        ]

        mistakes = layer2_samples[
            (layer2_samples['layer1_prediction'] != layer2_samples['final_prediction']) &
            (layer2_samples['final_prediction'] != layer2_samples['true_label'])
        ]

        print(f"Total Layer 2 samples: {len(layer2_samples)}")
        print(f"Correct corrections by Layer 2: {len(corrections)}")
        print(f"Incorrect changes by Layer 2: {len(mistakes)}")
        print(f"Net improvement rate: {(len(corrections) - len(mistakes)) / len(layer2_samples) * 100:.1f}%")

        if len(corrections) > 0:
            print(f"\nExamples of Layer 2 corrections:")
            for i, (_, row) in enumerate(corrections.head(3).iterrows()):
                print(f"   {i+1}. {row['url'][:60]}...")
                print(f"      Layer 1: {'Phishing' if row['layer1_prediction'] else 'Legitimate'} → "
                      f"Layer 2: {'Phishing' if row['final_prediction'] else 'Legitimate'} (Correct)")

        if len(mistakes) > 0:
            print(f"\nExamples of Layer 2 mistakes:")
            for i, (_, row) in enumerate(mistakes.head(3).iterrows()):
                print(f"   {i+1}. {row['url'][:60]}...")
                print(f"      Layer 1: {'Phishing' if row['layer1_prediction'] else 'Legitimate'} → "
                      f"Layer 2: {'Phishing' if row['final_prediction'] else 'Legitimate'} (Incorrect)")

    def save_evaluation_report(self, filename: str = None):
        """Save evaluation metrics to JSON file"""
        if not self.metrics:
            self.calculate_metrics()

        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"Evaluation report saved to: {filename}")
        return filename

def main():
    """Demo evaluation with sample data"""
    print("🔍 Hybrid Phishing Detection - Evaluation Demo")
    print("=" * 50)

    # Create sample evaluation data
    sample_data = {
        'url': [
            'https://www.google.com',
            'http://fake-bank.malicious.com',
            'https://www.amazon.com',
            'http://paypal-urgent.phishing.net',
            'https://www.facebook.com'
        ],
        'layer1_confidence': [0.1, 0.85, 0.2, 0.9, 0.15],
        'layer1_prediction': [0, 1, 0, 1, 0],
        'used_layer2': [False, True, False, False, False],
        'final_prediction': [0, 0, 0, 1, 0],
        'decision_maker': ['BiGRU', 'Phi3', 'BiGRU', 'BiGRU', 'BiGRU'],
        'true_label': [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(sample_data)
    evaluator = HybridModelEvaluator(results_df=df)

    # Run evaluation
    evaluator.calculate_metrics()
    evaluator.print_detailed_report()
    evaluator.analyze_layer2_effectiveness()

    # Save report
    evaluator.save_evaluation_report()

if __name__ == "__main__":
    main()
