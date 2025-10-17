import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import os
from datetime import datetime
import argparse
from hybrid_pipeline import HybridPhishingDetector

class BatchProcessor:
    """Batch processing for the hybrid phishing detection pipeline"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.detector = HybridPhishingDetector(
            bigru_model_path=self.config['models']['bigru']['model_path'],
            phi3_adapter_path=self.config['models']['phi3']['adapter_path'],
            bigru_threshold=self.config['pipeline']['bigru_threshold_low'],
            device=self.config['pipeline']['device']
        )

        # Create output directory
        os.makedirs(self.config['output']['output_directory'], exist_ok=True)

    def process_csv(self, input_file: str, url_column: str = 'url',
                   label_column: Optional[str] = None, output_file: Optional[str] = None) -> str:
        """Process URLs from a CSV file"""
        print(f"Processing CSV file: {input_file}")

        # Load data
        df = pd.read_csv(input_file)
        urls = df[url_column].astype(str).tolist()

        print(f"   Found {len(urls)} URLs to process")

        # Process in batches
        batch_size = self.config['pipeline']['batch_size']
        all_results = []

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i + batch_size]
            print(f"   Processing batch {i//batch_size + 1}/{(len(urls)-1)//batch_size + 1}")

            batch_results = self.detector.predict(batch_urls)
            all_results.extend(batch_results)

        # Create results dataframe
        results_df = pd.DataFrame(all_results)

        # Add original data
        for col in df.columns:
            if col != url_column:
                results_df[f'original_{col}'] = df[col].values[:len(results_df)]

        # Add evaluation metrics if labels are available
        if label_column and label_column in df.columns:
            true_labels = df[label_column].values[:len(results_df)]
            predictions = results_df['final_prediction'].values

            # Calculate metrics
            accuracy = np.mean(true_labels == predictions)
            precision = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(predictions == 1), 1)
            recall = np.sum((predictions == 1) & (true_labels == 1)) / max(np.sum(true_labels == 1), 1)
            f1 = 2 * precision * recall / max(precision + recall, 0.001)

            results_df['true_label'] = true_labels
            results_df['correct_prediction'] = true_labels == predictions

            print(f"\n Evaluation Metrics:")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.config['output']['output_directory'],
                f"batch_results_{timestamp}.csv"
            )

        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        # Print summary
        self.detector.print_stats()

        return output_file

    def process_url_list(self, urls: List[str], output_file: Optional[str] = None) -> str:
        """Process a list of URLs"""
        print(f"Processing {len(urls)} URLs")

        results = self.detector.predict(urls)
        results_df = pd.DataFrame(results)

        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.config['output']['output_directory'],
                f"url_list_results_{timestamp}.csv"
            )

        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

        self.detector.print_stats()

        return output_file

def create_sample_test_data():
    """Create sample test data for demonstration"""
    urls = [
        "https://www.google.com",
        "https://www.amazon.com",
        "https://www.facebook.com",
        "http://secure-banking-update.suspicious-site.com",
        "http://paypal-verification-required.fake-domain.net",
        "http://microsoft-security-alert.malicious.org",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "http://apple-id-suspended-verify.phishing-example.com",
        "http://google-security-check.fraudulent.site",
        "https://www.linkedin.com",
        "http://banking-urgent-action-required.scam.net",
        "https://www.youtube.com",
        "http://netflix-billing-issue.fake-streaming.org",
        "https://www.twitter.com"
    ]

    # Create labels
    labels = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]

    df = pd.DataFrame({
        'url': urls,
        'label': labels,
        'source': ['manual_test'] * len(urls)
    })

    df.to_csv('sample_test_data.csv', index=False)
    print("Sample test data created: sample_test_data.csv")

    return 'sample_test_data.csv'

def main():
    parser = argparse.ArgumentParser(description='Hybrid Phishing Detection Batch Processor')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file path')
    parser.add_argument('--url-column', '-u', type=str, default='url', help='Name of URL column')
    parser.add_argument('--label-column', '-l', type=str, help='Name of label column (for evaluation)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--create-sample', action='store_true', help='Create sample test data')
    parser.add_argument('--config', '-c', type=str, default='config.json', help='Configuration file path')

    args = parser.parse_args()

    if args.create_sample:
        sample_file = create_sample_test_data()
        args.input = sample_file
        args.label_column = 'label'

    if not args.input:
        print("No input file specified. Use --input or --create-sample")
        return

    # Initialize processor
    processor = BatchProcessor(args.config)

    # Process the file
    output_file = processor.process_csv(
        args.input,
        url_column=args.url_column,
        label_column=args.label_column,
        output_file=args.output
    )

    print(f"\nProcessing complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()
