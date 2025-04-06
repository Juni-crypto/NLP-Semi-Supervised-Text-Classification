import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        logger.info(f"Trainer initialized with device: {device}")
        
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        logger.info(f"Starting training epoch with {len(dataloader)} batches")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            labels, text = batch
            labels, text = labels.to(self.device), text.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(text)
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted_labels = predictions.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Log progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_accuracy = correct_predictions / total_predictions
                logger.info(f"Batch {batch_idx + 1}/{len(dataloader)} - Current Loss: {current_loss:.4f}, Current Accuracy: {current_accuracy:.4f}")
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        logger.info(f"Epoch completed - Final Loss: {avg_loss:.4f}, Final Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy
        
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                labels, text = batch
                labels, text = labels.to(self.device), text.to(self.device)
                
                predictions = self.model(text)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                predicted_labels = predictions.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy 