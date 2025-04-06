import torch
from torchtext.datasets import IMDB
import pandas as pd
import os
import logging
from tqdm import tqdm
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def save_imdb_data(split: str, output_dir: str = 'data'):
    """Save IMDB data for a given split to CSV"""
    logger.info(f"Processing {split} data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the iterator for the specified split
    iterator = IMDB(split=split)
    
    # Collect all samples
    data = []
    for label, text in tqdm(iterator, desc=f"Loading {split} data"):
        data.append({
            'label': label,
            'text': text
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Log initial distribution
    label_counts = df['label'].value_counts()
    logger.info(f"Initial label distribution in {split} data:\n{label_counts}")
    
    # For training data, ensure we have both positive and negative samples
    if split == 'train':
        # Separate positive and negative samples
        positive_samples = df[df['label'] == 2]
        negative_samples = df[df['label'] == 1]
        
        # Take equal number of samples from each class
        min_samples = min(len(positive_samples), len(negative_samples))
        logger.info(f"Taking {min_samples} samples from each class for training")
        
        # Randomly sample from each class
        positive_samples = positive_samples.sample(n=min_samples, random_state=42)
        negative_samples = negative_samples.sample(n=min_samples, random_state=42)
        
        # Combine and shuffle
        df = pd.concat([positive_samples, negative_samples])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Log final distribution
        label_counts = df['label'].value_counts()
        logger.info(f"Final label distribution in training data:\n{label_counts}")
    
    # Save to CSV
    output_file = os.path.join(output_dir, f'imdb_{split}.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {split} data to {output_file}")
    
    return df

def main():
    logger.info("Starting IMDB dataset download and processing...")
    
    # Process both splits
    train_df = save_imdb_data('train')
    test_df = save_imdb_data('test')
    
    # Print summary
    logger.info("\nDataset Summary:")
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    logger.info("\nTraining set label distribution:")
    logger.info(train_df['label'].value_counts())
    logger.info("\nTest set label distribution:")
    logger.info(test_df['label'].value_counts())
    
    logger.info("\nDataset processing completed!")

if __name__ == "__main__":
    main() 