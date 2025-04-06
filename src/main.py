import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader, Dataset
from data.data_loader import IMDBDataProcessor
from models.lstm_model import LSTMClassifier
from training.trainer import Trainer
import logging
from typing import List, Tuple
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

class IMDBDataset(Dataset):
    def __init__(self, data: List[Tuple[int, str]]):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def load_imdb_data(split: str, max_samples: int = None) -> List[Tuple[int, str]]:
    """Load IMDB data for a given split (train/test)"""
    logger.info(f"Loading {split} data...")
    data = []
    
    # Get the iterator for the specified split
    iterator = IMDB(split=split)
    
    # Collect all samples and convert labels to 0-based indexing
    for label, text in iterator:
        if label not in [1, 2]:
            logger.warning(f"Unexpected label {label} found in {split} data")
            continue
        # Convert label 1->0 (negative) and 2->1 (positive)
        data.append((label - 1, text))
    
    # Log distribution only for training data
    if split == 'test':  # This is actually our training data now
        label_counts = {0: 0, 1: 0}
        for label, _ in data:
            label_counts[label] += 1
        logger.info(f"Label distribution in training data: {label_counts}")
    
    return data

def create_balanced_dataset(data: List[Tuple[int, str]], max_samples_per_class: int = None, is_test: bool = False) -> IMDBDataset:
    """Create a balanced dataset with equal number of positive and negative samples"""
    logger.info("Creating balanced dataset...")
    
    # Separate positive and negative samples
    positive_samples = [(label, text) for label, text in data if label == 1]
    negative_samples = [(label, text) for label, text in data if label == 0]
    
    # Log initial counts only for training data
    if not is_test:
        logger.info(f"Found {len(positive_samples)} positive and {len(negative_samples)} negative samples")
    
    if not is_test and (len(positive_samples) == 0 or len(negative_samples) == 0):
        logger.error("Cannot create balanced dataset: missing samples")
        raise ValueError("Cannot create balanced dataset: missing samples")
    
    # For test data, just use all available samples
    if is_test:
        all_samples = data
    else:
        # Determine how many samples to take from each class
        if max_samples_per_class:
            samples_per_class = min(len(positive_samples), len(negative_samples), max_samples_per_class)
        else:
            samples_per_class = min(len(positive_samples), len(negative_samples))
        
        logger.info(f"Taking {samples_per_class} samples from each class")
        
        # Randomly sample from each class
        positive_samples = random.sample(positive_samples, samples_per_class)
        negative_samples = random.sample(negative_samples, samples_per_class)
        
        # Combine and shuffle
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)
    
    # Log final distribution only for training data
    if not is_test:
        label_counts = {0: 0, 1: 0}
        for label, _ in all_samples:
            label_counts[label] += 1
        logger.info(f"Final label distribution: {label_counts}")
    
    return IMDBDataset(all_samples)

def main():
    logger.info("Starting IMDB sentiment analysis training with swapped train/test data...")
    
    # Initialize data processor
    data_processor = IMDBDataProcessor()
    
    # Load data with swapped splits
    # Using test data as training data and train data as test data
    train_data = load_imdb_data('test')  # Using test data for training
    test_data = load_imdb_data('train')  # Using train data for testing
    
    # Create balanced datasets
    train_dataset = create_balanced_dataset(train_data, max_samples_per_class=12500)
    test_dataset = create_balanced_dataset(test_data, max_samples_per_class=2500, is_test=True)
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    data_processor.build_vocabulary(train_data)
    logger.info(f"Vocabulary size: {len(data_processor.vocab)}")
    
    # Model parameters
    VOCAB_SIZE = len(data_processor.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2  # Binary classification (0 or 1)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    # Initialize model and move to device
    model = LSTMClassifier(
        VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
        N_LAYERS, BIDIRECTIONAL, DROPOUT
    ).to(DEVICE)
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    
    # Initialize trainer and optimizer
    trainer = Trainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        collate_fn=data_processor.collate_batch, 
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        collate_fn=data_processor.collate_batch,
        shuffle=False
    )
    
    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(N_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS}")
        
        # Training
        train_loss, train_accuracy = trainer.train_epoch(train_dataloader, optimizer, criterion)
        logger.info(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        val_loss, val_accuracy = trainer.evaluate(test_dataloader, criterion)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # Save model state dict to CPU before saving to file
            model_state = model.state_dict()
            torch.save(model_state, 'model.pt')
            logger.info(f"Saved new best model with validation accuracy: {val_accuracy:.4f}")
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main() 