import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os
import json
from datetime import datetime
import warnings
import signal
import time
import importlib.util

# Add paths to import from both model directories
bigru_path = '/home/ghost-ed/Documents/Phishing model/BiGRU'
phi3_path = '/home/ghost-ed/Documents/Phishing model/Phi3'

sys.path.insert(0, bigru_path)
sys.path.insert(0, phi3_path)

# Load BiGRU model module
spec_model = importlib.util.spec_from_file_location("bigru_model", os.path.join(bigru_path, "model.py"))
bigru_model_module = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(bigru_model_module)
BiGRUPhishingDetector = bigru_model_module.BiGRUPhishingDetector

# Load preprocessor module
spec_preprocessor = importlib.util.spec_from_file_location("bigru_preprocessor", os.path.join(bigru_path, "preprocessor.py"))
preprocessor_module = importlib.util.module_from_spec(spec_preprocessor)
spec_preprocessor.loader.exec_module(preprocessor_module)
URLTextPreprocessor = preprocessor_module.URLTextPreprocessor

# Try to load Phi3 inference module
try:
    spec_phi3 = importlib.util.spec_from_file_location("phi3_inference", os.path.join(phi3_path, "inference_phi3_lora.py"))
    phi3_module = importlib.util.module_from_spec(spec_phi3)
    spec_phi3.loader.exec_module(phi3_module)
    Phi3LoRAInference = phi3_module.Phi3LoRAInference
except Exception as e:
    print(f"Warning: Could not load Phi3 module: {e}")
    Phi3LoRAInference = None

warnings.filterwarnings('ignore')

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

class HybridPhishingDetector:
    """
    Layer 1: BiGRU model for fast initial screening
    Layer 2: Phi3 LoRA model for detailed analysis of suspicious cases
    """

    def __init__(self,
                 bigru_model_path: str = "/home/ghost-ed/Documents/Phishing model/BiGRU/best_model.pth",
                 phi3_adapter_path: str = "/home/ghost-ed/Documents/Phishing model/Phi3/phi3_lora_finetuned_fast",
                 bigru_threshold: float = 0.3,
                 device: str = None,
                 phi3_timeout: int = 180):  # 3 minute timeout for Phi-3 loading

        self.bigru_model_path = bigru_model_path
        self.phi3_adapter_path = phi3_adapter_path
        self.bigru_threshold = bigru_threshold
        self.phi3_timeout = phi3_timeout
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Initializing Hybrid Phishing Detector on device: {self.device}")
        print(f"Using optimized Phi3 model: {phi3_adapter_path}")

        # Initialize components
        self.bigru_model = None
        self.bigru_preprocessor = None
        self.phi3_model = None

        # Statistics
        self.stats = {
            'total_processed': 0,
            'layer1_legitimate': 0,
            'layer1_suspicious': 0,
            'layer2_processed': 0,
            'final_phishing': 0,
            'final_legitimate': 0
        }

        self.load_models()

    def load_models(self):
        """Load both BiGRU and Phi3 models"""
        print("Loading Layer 1: BiGRU model...")
        self.load_bigru_model()

        print("Loading Layer 2: Phi3 model...")
        self.load_phi3_model()

        print("All models loaded successfully!")

    def load_bigru_model(self):
        """Load the trained BiGRU model and preprocessor"""
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

            print(f"   ✓ BiGRU model loaded (vocab_size: {vocab_size})")

        except Exception as e:
            print(f"   ✗ Error loading BiGRU model: {str(e)}")
            print("   → Using dummy BiGRU model for demonstration")
            self.create_dummy_bigru()

    def create_dummy_bigru(self):
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
        print("   ✓ Dummy BiGRU model created")

    def load_phi3_model(self):
        """Load the Phi3 LoRA model with timeout handling"""
        try:
            print(f"   Attempting to load Phi3 model (timeout: {self.phi3_timeout}s)...")

            # Set up timeout handler
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.phi3_timeout)

            start_time = time.time()
            self.phi3_model = Phi3LoRAInference(
                adapter_path=self.phi3_adapter_path,
                device="auto"
            )

            # Cancel the alarm
            signal.alarm(0)
            load_time = time.time() - start_time

            print(f"   Phi3 LoRA model loaded successfully! ({load_time:.1f}s)")

        except TimeoutException:
            print(f"   Timeout: Phi3 model loading exceeded {self.phi3_timeout}s")
            print("   Continuing with BiGRU-only mode")
            self.phi3_model = None
            signal.alarm(0)  # Cancel the alarm

        except Exception as e:
            print(f"   Error loading Phi3 model: {str(e)}")
            print("   Phi3 model not available, using BiGRU only")
            self.phi3_model = None
            signal.alarm(0)  # Cancel the alarm

    def predict_bigru(self, urls: List[str]) -> List[Dict]:
        """Layer 1: BiGRU predictions"""
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

                    results.append({
                        'url': url,
                        'bigru_confidence': confidence,
                        'bigru_prediction': int(confidence > 0.5),
                        'needs_phi3': confidence > self.bigru_threshold and confidence < (1 - self.bigru_threshold)
                    })

                except Exception as e:
                    print(f"Error processing URL in BiGRU: {url} - {str(e)}")
                    results.append({
                        'url': url,
                        'bigru_confidence': 0.5,
                        'bigru_prediction': 0,
                        'needs_phi3': True
                    })

        return results

    def predict_phi3(self, urls: List[str]) -> List[Dict]:
        """Layer 2: Phi3 predictions"""
        if self.phi3_model is None:
            print("Phi3 model not available, skipping Layer 2 analysis")
            return [{'url': url, 'phi3_prediction': 0, 'phi3_explanation': 'Model not available'} for url in urls]

        results = self.phi3_model.predict_batch(urls)
        return [{'url': r['url'], 'phi3_prediction': r['classification'], 'phi3_explanation': r['explanation']} for r in results]

    def predict(self, urls: List[str]) -> List[Dict]:
        """Main prediction pipeline"""
        if isinstance(urls, str):
            urls = [urls]

        print(f"\nProcessing {len(urls)} URLs through hybrid pipeline...")

        # Layer 1: BiGRU screening
        print("   Layer 1: BiGRU initial screening...")
        layer1_results = self.predict_bigru(urls)

        # Separate URLs that need Layer 2 analysis
        definite_legitimate = []
        definite_phishing = []
        suspicious_urls = []

        for result in layer1_results:
            if result['needs_phi3']:
                suspicious_urls.append(result['url'])
            elif result['bigru_prediction'] == 0:
                definite_legitimate.append(result)
            else:
                definite_phishing.append(result)

        print(f"      Definite legitimate: {len(definite_legitimate)}")
        print(f"      Definite phishing: {len(definite_phishing)}")
        print(f"      Suspicious (needs Layer 2): {len(suspicious_urls)}")

        # Layer 2: Phi3 analysis for suspicious URLs
        layer2_results = {}
        if suspicious_urls:
            print("   Layer 2: Phi3 detailed analysis...")
            phi3_predictions = self.predict_phi3(suspicious_urls)
            layer2_results = {r['url']: r for r in phi3_predictions}

        # Combine results
        final_results = []
        for result in layer1_results:
            url = result['url']
            final_result = {
                'url': url,
                'layer1_confidence': result['bigru_confidence'],
                'layer1_prediction': result['bigru_prediction'],
                'used_layer2': result['needs_phi3']
            }

            if result['needs_phi3'] and url in layer2_results:
                # Use Layer 2 (Phi3) result
                phi3_result = layer2_results[url]
                final_result.update({
                    'final_prediction': phi3_result['phi3_prediction'],
                    'layer2_explanation': phi3_result['phi3_explanation'],
                    'decision_maker': 'Phi3'
                })
            else:
                # Use Layer 1 (BiGRU) result
                final_result.update({
                    'final_prediction': result['bigru_prediction'],
                    'layer2_explanation': 'Not analyzed by Layer 2',
                    'decision_maker': 'BiGRU'
                })

            final_results.append(final_result)

        # Update statistics
        self.update_stats(final_results)

        return final_results

    def update_stats(self, results: List[Dict]):
        """Update processing statistics"""
        self.stats['total_processed'] += len(results)

        for result in results:
            if result['used_layer2']:
                self.stats['layer1_suspicious'] += 1
                self.stats['layer2_processed'] += 1
            elif result['layer1_prediction'] == 0:
                self.stats['layer1_legitimate'] += 1

            if result['final_prediction'] == 1:
                self.stats['final_phishing'] += 1
            else:
                self.stats['final_legitimate'] += 1

    def print_stats(self):
        """Print processing statistics"""
        print("\nProcessing Statistics:")
        print(f"   Total URLs processed: {self.stats['total_processed']}")
        print(f"   Layer 1 - Definite legitimate: {self.stats['layer1_legitimate']}")
        print(f"   Layer 1 - Suspicious (sent to Layer 2): {self.stats['layer1_suspicious']}")
        print(f"   Layer 2 - Processed by Phi3: {self.stats['layer2_processed']}")
        print(f"   Final - Phishing detected: {self.stats['final_phishing']}")
        print(f"   Final - Legitimate: {self.stats['final_legitimate']}")

        if self.stats['total_processed'] > 0:
            efficiency = (self.stats['total_processed'] - self.stats['layer2_processed']) / self.stats['total_processed'] * 100
            print(f"   Pipeline efficiency: {efficiency:.1f}% (URLs resolved by Layer 1)")

    def save_results(self, results: List[Dict], filename: str = None):
        """Save results to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_detection_results_{timestamp}.csv"

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"   Results saved to: {filename}")

        return filename

def main():
    # Initialize the hybrid detector
    detector = HybridPhishingDetector(
        bigru_threshold=0.3
    )

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

    print("Hybrid Phishing Detection Pipeline Demo")
    print("="*50)

    # Run predictions
    results = detector.predict(test_urls)

    # Display results
    print("\nDetection Results:")
    print("-" * 100)
    for i, result in enumerate(results, 1):
        status = "[PHISHING]" if result['final_prediction'] == 1 else "[LEGITIMATE]"
        layer = result['decision_maker']
        confidence = result['layer1_confidence']

        print(f"{i:2d}. {status} (by {layer}, conf: {confidence:.3f})")
        print(f"    URL: {result['url']}")
        if result['used_layer2'] and 'layer2_explanation' in result:
            explanation = result['layer2_explanation'][:100] + "..." if len(result['layer2_explanation']) > 100 else result['layer2_explanation']
            print(f"    Explanation: {explanation}")
        print()

    # Show statistics
    detector.print_stats()

    # Save results
    output_file = detector.save_results(results)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
