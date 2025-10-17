# Phishing Detection with Bi-GRU Neural Network

A deep learning-based phishing URL detection system using Bidirectional Gated Recurrent Units (Bi-GRU) with attention mechanism. This project aims to classify URLs as either phishing or legitimate by analyzing their textual patterns.

## 🎯 Overview

Phishing attacks remain one of the most prevalent cyber threats, where attackers create fraudulent websites to steal sensitive information. This project implements a neural network model that can detect phishing URLs by learning patterns in their structure and composition.

## ✨ Features

- **Bidirectional GRU Architecture**: Captures sequential patterns in URLs from both directions
- **Attention Mechanism**: Focuses on the most important parts of URLs for classification
- **Automatic Vocabulary Building**: Creates a vocabulary from URL tokens with configurable minimum frequency
- **Flexible Data Loading**: Supports multiple dataset formats with automatic column detection
- **Comprehensive Metrics**: Tracks accuracy, precision, recall, F1-score, and AUC
- **Training Visualization**: Plots training/validation loss and accuracy curves
- **Model Checkpointing**: Automatically saves the best model based on validation loss
- **GPU Support**: Utilizes CUDA when available for faster training

## 🏗️ Architecture

The model consists of the following components:

1. **Embedding Layer**: Converts URL tokens into dense vector representations (128-dimensional)
2. **Bidirectional GRU**: Two-layer Bi-GRU with 64 hidden units per direction
3. **Attention Mechanism**: Learns to focus on important URL components
4. **Classification Head**: Fully connected layers with dropout for binary classification

```
Input URL → Tokenization → Embedding → Bi-GRU → Attention → FC Layers → Output (0/1)
```

### Model Parameters

- Vocabulary Size: Dynamic (built from training data)
- Embedding Dimension: 128
- Hidden Dimension: 64 (per direction)
- Number of GRU Layers: 2
- Dropout Rate: 0.3
- Max Sequence Length: 200 tokens

## 📋 Requirements

### Python Dependencies

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
tqdm>=4.62.0
matplotlib>=3.4.0
kagglehub>=0.1.0
```

### System Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended for faster training)
- 4GB+ RAM
- Internet connection (for downloading dataset from Kaggle)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sowrin-paul/Phishing_model.git
   cd Phishing_model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch pandas numpy scikit-learn tqdm matplotlib kagglehub
   ```

4. **Verify CUDA availability** (optional)
   ```bash
   cd Bi-GRU
   python cuda_test.py
   ```

## 💻 Usage

### Training the Model

To train the model with default configuration:

```bash
cd Bi-GRU
python main.py
```

The script will:
1. Download the phishing dataset from Kaggle (if not already present)
2. Split data into train/validation/test sets (64%/16%/20%)
3. Build vocabulary from training URLs
4. Train the model for 20 epochs
5. Save the best model to `best_model.pth`
6. Display training/validation metrics and plots

### Configuration

You can modify the hyperparameters in `main.py`:

```python
config = {
    'max_length': 200,        # Maximum URL token sequence length
    'embedding_dim': 128,     # Embedding dimension
    'hidden_dim': 64,         # GRU hidden dimension
    'num_layers': 2,          # Number of GRU layers
    'dropout': 0.3,           # Dropout rate
    'batch_size': 32,         # Batch size for training
    'num_epochs': 20,         # Number of training epochs
    'min_freq': 2             # Minimum token frequency for vocabulary
}
```

### Using a Custom Dataset

To use your own dataset, modify the `load_phishing_data()` call in `main.py`:

```python
urls, labels = load_phishing_data(dataset_path='/path/to/your/dataset')
```

Your dataset should be a CSV file with:
- A column containing URLs (e.g., 'url', 'URL', 'website', 'domain')
- A column containing labels (e.g., 'label', 'target', 'class')
- Labels should be binary: 0 for legitimate, 1 for phishing

## 📊 Dataset

The default dataset is automatically downloaded from Kaggle:
- **Source**: [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- **Format**: CSV file with URLs and binary labels
- **Preprocessing**: Automatic detection of URL and label columns
- **Supported Label Formats**: 
  - Binary (0/1)
  - Text ('good'/'bad', 'legitimate'/'phishing', 'benign'/'phishing')

### Data Preprocessing

URLs are preprocessed as follows:
1. Remove protocol (http://, https://)
2. Remove 'www.' prefix
3. Tokenize by splitting on common delimiters (/, ., -, _, =, &, ?, %)
4. Convert to lowercase
5. Filter tokens with length > 1
6. Build vocabulary with configurable minimum frequency
7. Pad/truncate sequences to fixed length

## 📈 Model Performance

The model tracks the following metrics during training:

- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of true phishing predictions among all phishing predictions
- **Recall**: Proportion of actual phishing URLs correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under the ROC curve

Training progress is displayed in real-time with:
- Training loss per epoch
- Validation loss per epoch
- Validation metrics (accuracy, precision, recall, F1, AUC)

## 📁 Project Structure

```
Phishing_model/
│
├── Bi-GRU/
│   ├── main.py              # Main training script
│   ├── model.py             # Bi-GRU model architecture
│   ├── preprocessor.py      # Data loading and preprocessing
│   ├── train.py             # Training and validation logic
│   └── cuda_test.py         # CUDA availability test
│
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Key Components

### `model.py`
Defines the `BiGRUPhishingDetector` class with:
- Embedding layer with padding
- Bidirectional GRU layers
- Attention mechanism for focusing on important URL parts
- Fully connected classification layers

### `preprocessor.py`
Contains:
- `URLTextPreprocessor`: Tokenizes URLs and builds vocabulary
- `PhishingDataset`: PyTorch Dataset for loading data
- `load_phishing_data()`: Loads and preprocesses data from CSV

### `train.py`
Implements the `PhishingTrainer` class with:
- Training loop with gradient clipping
- Validation with multiple metrics
- Learning rate scheduling
- Model checkpointing
- Training history visualization

### `main.py`
Main entry point that:
- Loads and splits data
- Builds vocabulary
- Creates data loaders
- Initializes and trains the model
- Plots training history

## 🎓 Training Process

1. **Data Split**: 64% training, 16% validation, 20% test
2. **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-5
3. **Loss Function**: Binary Cross-Entropy (BCE)
4. **Learning Rate Scheduling**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
5. **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
6. **Early Stopping**: Best model saved based on lowest validation loss

## 🚀 Advanced Usage

### Loading a Trained Model

```python
import torch
from model import BiGRUPhishingDetector

# Initialize model with same config as training
model = BiGRUPhishingDetector(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=64,
    num_layers=2,
    dropout=0.3
)

# Load checkpoint
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Making Predictions

```python
from preprocessor import URLTextPreprocessor

# Preprocess URL
preprocessor = URLTextPreprocessor(max_length=200, min_freq=2)
# Load the vocabulary from training...

url = "http://suspicious-website.com/phishing-page"
sequence = preprocessor.text_to_sequence(url)
mask = preprocessor.create_attention_mask(sequence)

# Convert to tensors
sequence_tensor = torch.tensor([sequence], dtype=torch.long)
mask_tensor = torch.tensor([mask], dtype=torch.float)

# Make prediction
with torch.no_grad():
    output, attention_weights = model(sequence_tensor, mask_tensor)
    prediction = (output > 0.5).item()
    
print("Phishing" if prediction == 1 else "Legitimate")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

- **Sowrin Paul** - [sowrin-paul](https://github.com/sowrin-paul)

## 🙏 Acknowledgments

- Dataset provided by [Shashwat Work](https://www.kaggle.com/shashwatwork) on Kaggle
- Inspired by recent advances in NLP and sequence modeling for cybersecurity applications
- Built with PyTorch and scikit-learn

## 📚 References

If you use this code in your research, please consider citing relevant papers on:
- Bidirectional RNNs and GRUs
- Attention mechanisms in neural networks
- Phishing detection using deep learning

## 🐛 Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in config or use CPU

**Issue**: Dataset download fails
- **Solution**: Ensure stable internet connection and valid Kaggle credentials

**Issue**: Low accuracy
- **Solution**: Try increasing num_epochs, adjusting learning rate, or collecting more training data

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## 🔮 Future Improvements

- [ ] Add support for ensemble models
- [ ] Implement cross-validation
- [ ] Add real-time URL scanning API
- [ ] Support for additional features (domain age, WHOIS data)
- [ ] Integration with web browser extension
- [ ] Multi-class classification (different types of phishing)
- [ ] Transfer learning from pre-trained language models

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the repository maintainer.

---

**Note**: This model is for educational and research purposes. For production use, consider additional security measures and regular model updates with new phishing patterns.
