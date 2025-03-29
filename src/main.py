import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
from data.data_loader import IMDBDataProcessor
from models.lstm_model import LSTMClassifier
from training.trainer import Trainer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting IMDB sentiment analysis training...")
    
    # Initialize data processor
    logger.info("Initializing data processor...")
    data_processor = IMDBDataProcessor()
    
    # Load IMDB dataset
    logger.info("Loading IMDB dataset...")
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    logger.info(f"Dataset loaded successfully. Training set size: {len(list(train_iter))}, Test set size: {len(list(test_iter))}")
    
    # Build vocabulary
    logger.info("Building vocabulary from training data...")
    data_processor.build_vocabulary(train_iter)
    logger.info(f"Vocabulary built successfully. Vocabulary size: {len(data_processor.vocab)}")
    
    # Model parameters
    VOCAB_SIZE = len(data_processor.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    logger.info("Initializing model with parameters:")
    logger.info(f"Vocabulary Size: {VOCAB_SIZE}")
    logger.info(f"Embedding Dimension: {EMBEDDING_DIM}")
    logger.info(f"Hidden Dimension: {HIDDEN_DIM}")
    logger.info(f"Number of Layers: {N_LAYERS}")
    logger.info(f"Bidirectional: {BIDIRECTIONAL}")
    logger.info(f"Dropout: {DROPOUT}")
    
    # Initialize model
    model = LSTMClassifier(
        VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
        N_LAYERS, BIDIRECTIONAL, DROPOUT
    )
    logger.info("Model initialized successfully")
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 5
    
    logger.info("Setting up training parameters:")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Number of Epochs: {N_EPOCHS}")
    
    # Initialize trainer and optimizer
    logger.info("Initializing trainer and optimizer...")
    trainer = Trainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = DataLoader(
        list(train_iter), batch_size=BATCH_SIZE,
        collate_fn=data_processor.collate_batch,
        shuffle=True
    )
    logger.info(f"Data loader created with {len(train_dataloader)} batches")
    
    # Training loop
    logger.info("Starting training loop...")
    for epoch in range(N_EPOCHS):
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS} starting...")
        loss, accuracy = trainer.train_epoch(train_dataloader, optimizer, criterion)
        logger.info(f"Epoch {epoch+1}/{N_EPOCHS} completed")
        logger.info(f"Average Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 