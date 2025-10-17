import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from collections import Counter
from urllib.parse import urlparse
import os
import kagglehub


class URLTextPreprocessor:
    def __init__(self, max_length=200, min_freq=2):
        self.max_length = max_length
        self.min_freq = min_freq
        self.vocab = {}
        self.vocab_size = 0
        self.word_to_idx = {}
        self.idx_to_word = {}

    def clean_url(self, url):
        url = re.sub(r'https?://', '', url)
        url = re.sub(r'www\.', '', url)

        tokens = re.split(r'[./\-_=&?%]', url.lower())

        tokens = [token for token in tokens if len(token) > 1]

        return tokens

    def build_vocab(self, texts):
        all_tokens = []
        for text in texts:
            if isinstance(text, str):
                tokens = self.clean_url(text)
                all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2

        for token, count in token_counts.items():
            if count >= self.min_freq:
                self.word_to_idx[token] = idx
                idx += 1

        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)

        print(f"Vocabulary size: {self.vocab_size}")

    def text_to_sequence(self, text):
        if isinstance(text, str):
            tokens = self.clean_url(text)
        else:
            tokens = []

        sequence = [self.word_to_idx.get(token, 1) for token in tokens]

        # Pad or truncate
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([0] * (self.max_length - len(sequence)))

        return sequence

    def create_attention_mask(self, sequence):
        return [1 if token != 0 else 0 for token in sequence]


class PhishingDataset(Dataset):
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        sequence = self.preprocessor.text_to_sequence(text)
        mask = self.preprocessor.create_attention_mask(sequence)

        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }

def load_phishing_data(dataset_path=None, hf_dataset_id="ealvaradob/phishing-dataset", split_name="train"):
    import pandas as pd
    import os

    if dataset_path is None:
        try:
            from huggingface_hub import hf_hub_download
            import json
        except ImportError:
            raise ImportError("Please install the Hugging Face Hub library: `pip install huggingface_hub`")

        print(f"Loading dataset from Hugging Face: {hf_dataset_id}")

        # Download the urls.json file (contains URLs without spam messages)
        print("   Downloading urls.json (URL-only data, no spam messages)...")
        file_path = hf_hub_download(
            repo_id=hf_dataset_id,
            filename='urls.json',
            repo_type='dataset'
        )

        # Load JSON data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"   Successfully loaded {len(df):,} URLs from ealvaradob/phishing-dataset")

    else:
        # Local CSV path (existing behavior)
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")

        csv_file = csv_files[0]
        full_path = os.path.join(dataset_path, csv_file)
        print(f"Loading data from: {full_path}")
        df = pd.read_csv(full_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Detect URL and label columns
    url_column = None
    label_column = None

    # For ealvaradob dataset, the columns are 'text' and 'label'
    url_candidates = ['text', 'url', 'URL', 'website', 'domain', 'link']
    for col in df.columns:
        if col.lower() in [c.lower() for c in url_candidates]:
            url_column = col
            break

    label_candidates = ['label', 'Label', 'target', 'class', 'result', 'status']
    for col in df.columns:
        if col.lower() in [c.lower() for c in label_candidates]:
            label_column = col
            break

    if url_column is None:
        url_column = df.columns[0]
        print(f"Auto-detected URL column: {url_column}")

    if label_column is None:
        label_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        print(f"Auto-detected label column: {label_column}")

    # Keep ONLY URL and label columns - exclude any message/text content
    df = df[[url_column, label_column]]
    print(f"Using only URL and label columns (no spam message content)")

    df = df.dropna(subset=[url_column, label_column])
    df[url_column] = df[url_column].astype(str)

    unique_labels = df[label_column].unique()
    print(f"Unique labels found: {unique_labels}")

    # Handle different label formats
    if set(unique_labels) == {0, 1}:
        pass
    elif set(unique_labels) == {'good', 'bad'}:
        df[label_column] = df[label_column].map({'good': 0, 'bad': 1})
    elif set(unique_labels) == {'legitimate', 'phishing'}:
        df[label_column] = df[label_column].map({'legitimate': 0, 'phishing': 1})
    elif set(unique_labels) == {'benign', 'phishing'}:
        df[label_column] = df[label_column].map({'benign': 0, 'phishing': 1})
    else:
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        df = df.dropna(subset=[label_column])
        df[label_column] = df[label_column].astype(int)

    final_labels = df[label_column].unique()
    print(f"Final labels after conversion: {final_labels}")
    print("Label distribution:")
    print(df[label_column].value_counts())
    print(f"\nFirst few URLs:")
    print(df[url_column].head())

    return df[url_column].tolist(), df[label_column].tolist()
