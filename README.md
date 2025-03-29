# Semi-Supervised Text Classification with LSTM

This project implements a semi-supervised text classification model using LSTM (Long Short-Term Memory) and Virtual Adversarial Training (VAT) techniques. The model is trained on the IMDB dataset for sentiment analysis.

## Project Structure

```
project/
├── data/               # For storing datasets
├── src/               
│   ├── data/          # Data processing scripts
│   ├── models/        # Model implementations
│   ├── training/      # Training scripts
│   └── utils/         # Utility functions
├── config/            # Configuration files
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py
```

## Model Architecture

- LSTM-based text classifier
- Bidirectional LSTM layers
- Dropout for regularization
- Virtual Adversarial Training (VAT) for semi-supervised learning

## Features

- Text preprocessing and tokenization
- Vocabulary building
- Batch processing with padding
- Training with progress monitoring
- Accuracy and loss tracking

## Next Steps

1. Implement Virtual Adversarial Training (VAT)
2. Add validation during training
3. Implement model evaluation
4. Add configuration file support
5. Add model checkpointing 