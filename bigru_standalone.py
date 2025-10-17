import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os
import json
from datetime import datetime
import warnings

# Add paths to import from both model directories
sys.path.append('/BiGRU')

from .Bi-GRU .model import BiGRUPhishingDetector
from preprocessor import URLTextPreprocessor

warnings.filterwarnings('ignore')

class BiGRUPhishingDetector_Standalone:
    """
    Standalone BiGRU phishing detection system
    Fast and reliable single-layer detection
    """

    def __init__(self,
                 bigru_model_path: str = "/home/ghost-ed/Documents/Phishing model/BiGRU/best_model.pth",
                 device: str = None):

        self.bigru_model_path = bigru_model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing BiGRU Phishing Detector on device: {self.device}")

        # Initialize components
        self.bigru_model = None
        self.bigru_preprocessor = None

        # Statistics
        self.stats = {
            'total_processed': 0,
            'phishing_detected': 0,
            'legitimate_detected': 0
        }

        self.load_model()

    def load_model(self):
        """Load BiGRU model"""
        print("Loading BiGRU model...")
        try:
            # Load model checkpoint
            checkpoint = torch.load(self.bigru_model_path, map_location=self.device)

            # Initialize preprocessor
            self.bigru_preprocessor = URLTextPreprocessor(max_length=200, min_freq=2)

            dummy_urls = [
                "http://example.com", "https://google.com", "http://phishing-site.com",
                "https://secure-bank.com", "http://suspicious.link"
            ]
            self.bigru_preprocessor.build_vocab(dummy_urls)

            # Initialize model architecture
            vocab_size = self.bigru_preprocessor.vocab_size
            self.bigru_model = BiGRUPhishingDetector(
                vocab_size=vocab_size,
                embedding_dim=128,
                hidden_dim=64,
                num_layers=2,
                dropout=0.3
            ).to(self.device)

            # Load trained weights
            self.bigru_model.load_state_dict(checkpoint['model_state_dict'])
            self.bigru_model.eval()

            print(f"   BiGRU model loaded successfully (vocab_size: {vocab_size})")

        except Exception as e:
            print(f"   Error loading BiGRU model: {str(e)}")
            print("   Using dummy BiGRU model for demonstration")
            self.create_dummy_model()

    def create_dummy_model(self):
        """Create a dummy BiGRU model if the trained model is not available"""
        self.bigru_preprocessor = URLTextPreprocessor(max_length=200, min_freq=2)
        dummy_urls = [
            "http://example.com", "https://google.com", "http://phishing-site.com",
            "https://secure-bank.com", "http://suspicious.link", "https://facebook.com",
            "http://malicious.site", "https://amazon.com", "http://fake-bank.net"
        ]
        self.bigru_preprocessor.build_vocab(dummy_urls)

        self.bigru_model = BiGRUPhishingDetector(
            vocab_size=self.bigru_preprocessor.vocab_size,
            embedding_dim=128,
            hidden_dim=64,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        # Initialize with random weights
        self.bigru_model.eval()
        print("   Dummy BiGRU model created")

    def predict(self, urls: List[str]) -> List[Dict]:
        """Make predictions for URLs"""
        if isinstance(urls, str):
            urls = [urls]

        print(f"\nProcessing {len(urls)} URLs...")
        results = []

        with torch.no_grad():
            for url in urls:
                try:
                    # Preprocess URL
                    sequence = self.bigru_preprocessor.text_to_sequence(url)
                    mask = self.bigru_preprocessor.create_attention_mask(sequence)

                    # Convert to tensors
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device)
                    mask_tensor = torch.tensor([mask], dtype=torch.float).to(self.device)

                    # Predict
                    output, attention_weights = self.bigru_model(sequence_tensor, mask_tensor)
                    confidence = float(output.cpu().numpy()[0])
                    prediction = int(confidence > 0.5)

                    results.append({
                        'url': url,
                        'confidence': confidence,
                        'prediction': prediction,
                        'status': 'phishing' if prediction == 1 else 'legitimate'
                    })

                except Exception as e:
                    print(f"Error processing URL: {url} - {str(e)}")
                    results.append({
                        'url': url,
                        'confidence': 0.5,
                        'prediction': 0,
                        'status': 'error'
                    })

        # Update statistics
        self.update_stats(results)
        return results

    def update_stats(self, results: List[Dict]):
        """Update processing statistics"""
        self.stats['total_processed'] += len(results)
        for result in results:
            if result['prediction'] == 1:
                self.stats['phishing_detected'] += 1
            else:
                self.stats['legitimate_detected'] += 1

    def print_stats(self):
        """Print processing statistics"""
        print("\nProcessing Statistics:")
        print(f"   Total URLs processed: {self.stats['total_processed']}")
        print(f"   Phishing detected: {self.stats['phishing_detected']}")
        print(f"   Legitimate detected: {self.stats['legitimate_detected']}")

    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bigru_detection_results_{timestamp}.csv"

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"   Results saved to: {filename}")
        return filename

def main():
    # Initialize the detector
    detector = BiGRUPhishingDetector_Standalone()

    # Test URLs
    test_urls = [
        "https://www.google.com",
        "http://secure-banking-login-verification.suspicious-domain.com",
        "https://www.amazon.com",
        "http://paypal-security-update-urgent.fake-domain.net",
        "https://www.facebook.com",
        "http://microsoft-account-suspended-click-here.malicious.org",
        "https://www.github.com",
        "http://apple-id-locked-verify-now.phishing-site.com"
    ]

    print("BiGRU Phishing Detection Demo")
    print("="*40)

    # Run predictions
    results = detector.predict(test_urls)

    # Display results
    print("\nDetection Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        status = "[PHISHING]" if result['prediction'] == 1 else "[LEGITIMATE]"
        confidence = result['confidence']

        print(f"{i:2d}. {status} (confidence: {confidence:.3f})")
        print(f"    URL: {result['url']}")
        print()

    # Show statistics
    detector.print_stats()

    # Save results
    output_file = detector.save_results(results)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
