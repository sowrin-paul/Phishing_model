# Phishing URL Detection using Bi-GRU with Attention

A deep learning-based phishing detection system that uses Bidirectional Gated Recurrent Units (Bi-GRU) with attention mechanism to identify phishing URLs. The model analyzes URL patterns and structures to classify websites as legitimate or phishing attempts.

## 🚀 Features

- **Bidirectional GRU Architecture**: Captures sequential patterns in URLs from both directions
- **Attention Mechanism**: Focuses on important URL components for better classification
- **Automatic Dataset Handling**: Downloads and processes phishing datasets from Kaggle
- **Comprehensive Training Pipeline**: Includes data preprocessing, training, validation, and evaluation
- **Performance Metrics**: Tracks accuracy, precision, recall, F1-score, and AUC-ROC
- **Training Visualization**: Plots training/validation loss and accuracy curves
- **GPU Support**: Automatically detects and uses CUDA-enabled GPUs when available

## 📋 Requirements

### Python Dependencies

```
torch>=1.10.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
tqdm>=4.62.0
kagglehub>=0.1.0
sympy
```

### System Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended for faster training)
- 4GB+ RAM
- Internet connection (for downloading datasets)

## 🔧 Installation

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
pip install torch pandas numpy scikit-learn matplotlib tqdm kagglehub sympy
```

4. **Configure Kaggle API** (for automatic dataset download)
- Create a Kaggle account at https://www.kaggle.com
- Go to Account settings and create a new API token
- Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Username>\.kaggle\` (Windows)

## 📊 Dataset

The project uses the [Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset) from Kaggle. The dataset is automatically downloaded when you run the training script.

**Dataset Features:**
- Contains URLs with binary labels (legitimate: 0, phishing: 1)
- Supports multiple label formats (good/bad, legitimate/phishing, benign/phishing)
- Automatic column detection for URLs and labels

## 🏗️ Model Architecture

### BiGRU with Attention

The model consists of the following components:

1. **Embedding Layer**
   - Converts URL tokens into dense vector representations
   - Dimension: 128 (configurable)
   - Includes padding for variable-length sequences

2. **Bidirectional GRU**
   - 2 layers by default
   - Hidden dimension: 64 (configurable)
   - Processes URL sequences in both forward and backward directions
   - Dropout: 0.3 for regularization

3. **Attention Mechanism**
   - Learns to focus on important URL components
   - Uses masked softmax to ignore padding tokens
   - Produces weighted representation of GRU outputs

4. **Classification Layers**
   - Fully connected layers (128 → 32 → 1)
   - ReLU activation
   - Dropout for regularization
   - Sigmoid output for binary classification

### Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| max_length | 200 | Maximum URL token sequence length |
| embedding_dim | 128 | Embedding dimension |
| hidden_dim | 64 | GRU hidden dimension |
| num_layers | 2 | Number of GRU layers |
| dropout | 0.3 | Dropout probability |
| batch_size | 32 | Training batch size |
| num_epochs | 20 | Number of training epochs |
| learning_rate | 0.001 | Initial learning rate |
| min_freq | 2 | Minimum token frequency for vocabulary |

## 🎯 Usage

### Training the Model

Navigate to the `Bi-GRU` directory and run the main script:

```bash
cd Bi-GRU
python main.py
```

The script will:
1. Download the dataset from Kaggle (first run only)
2. Preprocess URLs and build vocabulary
3. Split data into train/validation/test sets (64%/16%/20%)
4. Train the model for 20 epochs
5. Save the best model as `best_model.pth`
6. Display training metrics and plots

### Testing CUDA Availability

To check if your system can use GPU acceleration:

```bash
cd Bi-GRU
python cuda_test.py
```

### Custom Configuration

You can modify hyperparameters in `main.py`:

```python
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
```

### Using a Pre-trained Model

```python
import torch
from model import BiGRUPhishingDetector

# Load model
checkpoint = torch.load('best_model.pth')
model = BiGRUPhishingDetector(vocab_size=vocab_size, ...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
# ... (preprocess URL and create input tensor)
output, attention_weights = model(input_tensor, mask)
prediction = (output > 0.5).float()
```

## 📁 Project Structure

```
Phishing_model/
├── Bi-GRU/
│   ├── main.py              # Main training script
│   ├── model.py             # BiGRU model architecture
│   ├── preprocessor.py      # Data loading and preprocessing
│   ├── train.py             # Training and evaluation logic
│   └── cuda_test.py         # CUDA availability checker
├── .gitignore               # Git ignore file
└── README.md                # Project documentation (this file)
```

### File Descriptions

- **main.py**: Entry point for training. Handles data loading, model initialization, training loop, and saves results
- **model.py**: Defines the BiGRUPhishingDetector neural network with attention mechanism
- **preprocessor.py**: Contains URLTextPreprocessor, PhishingDataset, and data loading functions
- **train.py**: PhishingTrainer class with training/validation logic and metrics tracking
- **cuda_test.py**: Simple utility to check CUDA availability

## 📈 Training Process

### Data Preprocessing

1. **URL Cleaning**
   - Removes HTTP/HTTPS protocols
   - Removes 'www.' prefix
   - Tokenizes URLs by splitting on delimiters (/, ., -, _, =, &, ?, %)
   - Converts to lowercase
   - Filters tokens with length > 1

2. **Vocabulary Building**
   - Creates token-to-index mapping
   - Filters tokens by minimum frequency (default: 2)
   - Adds special tokens: `<PAD>` (0), `<UNK>` (1)

3. **Sequence Creation**
   - Converts URLs to integer sequences
   - Pads/truncates to maximum length (200)
   - Creates attention masks for padding tokens

### Training Configuration

- **Loss Function**: Binary Cross-Entropy (BCE)
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Gradient Clipping**: Max norm of 1.0
- **Early Stopping**: Saves best model based on validation loss

### Evaluation Metrics

The model tracks the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of correctly identified phishing URLs
- **Recall**: Proportion of actual phishing URLs detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 🎨 Visualization

Training progress is visualized with two plots:
1. **Loss Curves**: Training and validation loss over epochs
2. **Accuracy Curve**: Validation accuracy over epochs

## 🔍 How It Works

1. **URL Tokenization**: URLs are split into meaningful tokens (domain parts, paths, parameters)
2. **Embedding**: Tokens are converted into dense vector representations
3. **Bidirectional Processing**: GRU processes the sequence from both directions to capture context
4. **Attention**: The model learns which parts of the URL are most important for classification
5. **Classification**: The attended representation is passed through fully connected layers for final prediction

## 🚀 Performance Tips

- **Use GPU**: Training is significantly faster with CUDA-enabled GPU
- **Increase Batch Size**: If you have sufficient GPU memory, increase batch size for faster training
- **Adjust Epochs**: Monitor validation metrics and adjust the number of epochs to avoid overfitting
- **Tune Hyperparameters**: Experiment with embedding dimensions, hidden dimensions, and dropout rates

## 📝 Example Output

```
Loading phishing dataset...
Downloading dataset from Kaggle...
Dataset downloaded to: /path/to/dataset
Total samples loaded: 10000
Phishing samples: 5000
Legitimate samples: 5000
Train samples: 6400
Val samples: 1600
Test samples: 2000
Building vocabulary...
Vocabulary size: 8543
Initializing model...
Model parameters: 1,234,567

Epoch 1/20
Training: 100%|██████████| 200/200 [00:45<00:00,  4.42it/s]
Validation: 100%|██████████| 50/50 [00:05<00:00,  9.12it/s]
Train Loss: 0.4521
Val Loss: 0.3876
Accuracy: 0.8543
Precision: 0.8621
Recall: 0.8432
F1: 0.8525
AUC: 0.9234
New best model saved with val_loss: 0.3876
...
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas

- Add support for more datasets
- Implement additional model architectures (LSTM, Transformer, etc.)
- Add real-time URL scanning API
- Create web interface for predictions
- Improve preprocessing techniques
- Add more evaluation metrics and visualizations

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: [Shashwat Work's Web Page Phishing Detection Dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset)
- PyTorch framework for deep learning implementation
- Kaggle for hosting the dataset

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## 🔮 Future Work

- [ ] Implement ensemble methods with multiple models
- [ ] Add explainability features (SHAP, LIME)
- [ ] Create REST API for real-time predictions
- [ ] Deploy model as a web service
- [ ] Add support for multi-class classification
- [ ] Implement transfer learning from pre-trained models
- [ ] Create browser extension for real-time URL checking
- [ ] Add adversarial training for robustness

---

**Note**: This is a research/educational project. For production use, ensure thorough testing and validation with your specific use case and data.
