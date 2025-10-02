import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import BiGRUPhishingDetector
from preprocessor import URLTextPreprocessor, PhishingDataset, load_phishing_data
from train import PhishingTrainer


def main():

    config = {
        'max_length': 200,
        'embedding_dim': 128,
        'hidden_dim': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'batch_size': 32,
        'num_epochs': 20,
        'min_freq': 2
    }

    print("Loading phishing dataset...")
    urls, labels = load_phishing_data()

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

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
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

    # Train model
    print("Starting training...")
    trainer = PhishingTrainer(model, train_loader, val_loader)
    trainer.train(num_epochs=config['num_epochs'])

    # Plot training history
    trainer.plot_training_history()

    print("Training completed!")


if __name__ == "__main__":
    main()