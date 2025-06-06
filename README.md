# RADAR#: Radicalization Analysis and Detection in Arabic Resources

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/tensorflow-2.8+-orange.svg)](https://www.tensorflow.org/)

## Overview

RADAR# is a deep learning framework for detecting and analyzing radicalization content in Arabic social media and text. The framework combines CNN-BiLSTM architecture with attention mechanisms and transformer models to achieve state-of-the-art performance in identifying various categories of radicalization indicators.

## Key Features

- **Hybrid Architecture**: Combines CNN-BiLSTM with attention mechanisms and transformer models
- **Multi-category Classification**: Detects multiple categories of radicalization (Explicit, Implicit, Borderline, Propaganda)
- **Arabic-specific Processing**: Specialized preprocessing for Modern Standard Arabic and dialectal variations
- **Interpretability**: Attention visualization for model decision explanation
- **Comprehensive Evaluation**: Detailed error analysis and performance metrics

## Repository Structure

```
radar_project/
├── data/                      # Dataset files and documentation
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Processed dataset files
│   └── README.md              # Dataset documentation
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── preprocessing.py   # Arabic text preprocessing
│   │   ├── dataset.py         # Dataset handling
│   │   └── augmentation.py    # Data augmentation
│   ├── models/                # Model architecture
│   │   ├── cnn_bilstm.py      # CNN-BiLSTM implementation
│   │   ├── attention.py       # Attention mechanism
│   │   ├── transformer.py     # Transformer integration
│   │   └── ensemble.py        # Ensemble model
│   ├── training/              # Training utilities
│   │   ├── trainer.py         # Model training
│   │   ├── optimizer.py       # Optimizer configurations
│   │   └── hyperparameters.py # Hyperparameter optimization
│   ├── evaluation/            # Evaluation utilities
│   │   ├── metrics.py         # Evaluation metrics
│   │   ├── error_analysis.py  # Error analysis
│   │   └── interpretability.py # Interpretability visualization
│   └── generate_figures.py    # Script to generate all figures
├── results/                   # Results and outputs
│   └── figures/               # Generated figures
├── notebooks/                 # Jupyter notebooks for experiments
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup script
├── README.md                  # Main repository documentation
└── LICENSE                    # License file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/username/radar-project.git
cd radar-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```python
from src.data.preprocessing import ArabicPreprocessor

# Initialize preprocessor
preprocessor = ArabicPreprocessor(
    normalize=True,
    remove_diacritics=True,
    remove_stopwords=False
)

# Preprocess text
processed_text = preprocessor.preprocess_text("نص عربي للمعالجة")
```

### Model Training

```python
from src.models.cnn_bilstm import CNNBiLSTM
from src.training.trainer import ModelTrainer

# Initialize model
model = CNNBiLSTM(
    vocab_size=30000,
    embedding_dim=300,
    max_sequence_length=100,
    lstm_units=128,
    num_filters=128,
    dropout_rate=0.5,
    num_classes=5
).build_model()

# Initialize trainer
trainer = ModelTrainer(
    model=model,
    model_name="radar_cnn_bilstm",
    output_dir="results/models"
)

# Train model
trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
```

### Evaluation and Visualization

```python
from src.evaluation.metrics import calculate_metrics, plot_confusion_matrix
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.interpretability import AttentionVisualizer

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred, y_prob)

# Plot confusion matrix
plot_confusion_matrix(
    y_true, y_pred,
    class_names=['Non-radical', 'Explicit', 'Implicit', 'Borderline', 'Propaganda'],
    save_path="results/figures/confusion_matrix.png"
)

# Analyze errors
analyzer = ErrorAnalyzer(y_true, y_pred, y_prob, texts)
error_distribution = analyzer.get_error_distribution()

# Visualize attention weights
visualizer = AttentionVisualizer(texts, attention_weights)
visualizer.visualize_attention(
    sample_idx=0,
    save_path="results/figures/attention_visualization.png"
)
```

### Generate All Figures

```bash
python src/generate_figures.py
```

## Results

The RADAR# framework achieves state-of-the-art performance on Arabic radicalization detection:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| BERT-Arabic | 82% | 81% | 83% |
| AraBERT | 85% | 84% | 86% |
| CNN-LSTM | 79% | 77% | 80% |
| MarBERT | 87% | 86% | 88% |
| BiLSTM-Attention | 83% | 82% | 84% |
| **RADAR#** | **92%** | **91%** | **93%** |

## Citation

If you use this code or dataset in your research, please cite:

```
@article{radar2025,
  title={RADAR#: A Deep Learning Framework for Radicalization Detection in Arabic Social Media},
  author={[Emad M. Al-Shawakfa, Anas M.R. Alsobeh, Sahar Omari, Amani Shatnawi]},
  journal={[Information, MDPI]},
  year={2025},
  volume={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

