import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from model import BiGRUPhishingDetector
from preprocessor import URLTextPreprocessor, PhishingDataset, load_phishing_data
from train import PhishingTrainer


def main():
    # Optimized configuration for faster training
    config = {
        'max_length': 150,  # Reduced from 200 for faster processing
        'embedding_dim': 128,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 128,  # Increased from 32 for faster training
        'num_epochs': 15,  # Reduced from 20 - OneCycleLR helps converge faster
        'min_freq': 2,
        'num_workers': 4,  # Parallel data loading
        'sample_size': 100000  # Limit dataset size for faster training
    }

    print("Loading phishing dataset...")
    urls, labels = load_phishing_data()

    # Sample dataset for faster training if it's too large
    if len(urls) > config['sample_size']:
        print(f"Sampling {config['sample_size']} URLs from {len(urls)} total...")
        indices = list(range(len(urls)))
        import random
        random.seed(42)
        random.shuffle(indices)
        sampled_indices = indices[:config['sample_size']]
        urls = [urls[i] for i in sampled_indices]
        labels = [labels[i] for i in sampled_indices]

    print(f"Total samples loaded: {len(urls)}")
    print(f"Phishing samples: {sum(labels)}")
    print(f"Legitimate samples: {len(labels) - sum(labels)}")

    # Split data
    train_urls, test_urls, train_labels, test_labels = train_test_split(
        urls, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_urls, val_urls, train_labels, val_labels = train_test_split(
        train_urls, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    print(f"Train samples: {len(train_urls)}")
    print(f"Val samples: {len(val_urls)}")
    print(f"Test samples: {len(test_urls)}")

    # Preprocessing
    print("Building vocabulary...")
    preprocessor = URLTextPreprocessor(
        max_length=config['max_length'],
        min_freq=config['min_freq']
    )
    preprocessor.build_vocab(train_urls)

    # Create datasets
    train_dataset = PhishingDataset(train_urls, train_labels, preprocessor)
    val_dataset = PhishingDataset(val_urls, val_labels, preprocessor)
    test_dataset = PhishingDataset(test_urls, test_labels, preprocessor)

    # Create optimized data loaders with pin_memory and num_workers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=(device == 'cuda'),
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=(device == 'cuda'),
        persistent_workers=True if config['num_workers'] > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'] * 2,
        shuffle=False
    )

    # Initialize model
    print("Initializing model...")
    model = BiGRUPhishingDetector(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using mixed precision training: YES")
        print(f"Gradient accumulation steps: 2 (effective batch size: {config['batch_size'] * 2})")

    # Train model
    print("Starting training...")
    trainer = PhishingTrainer(model, train_loader, val_loader)
    trainer.train(num_epochs=config['num_epochs'])

    # Plot training history
    trainer.plot_training_history()

    print("Training completed!")


if __name__ == "__main__":
    main()