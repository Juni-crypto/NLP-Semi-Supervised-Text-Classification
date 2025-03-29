import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from typing import List, Tuple, Iterator
import logging

logger = logging.getLogger(__name__)

class IMDBDataProcessor:
    def __init__(self, max_length: int = 256):
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = max_length
        self.vocab = None
        logger.info(f"Initialized IMDBDataProcessor with max_length={max_length}")
        
    def yield_tokens(self, data_iter: Iterator) -> List[str]:
        for _, text in data_iter:
            yield self.tokenizer(text)
            
    def build_vocabulary(self, train_iter: Iterator):
        logger.info("Building vocabulary from training data...")
        self.vocab = build_vocab_from_iterator(
            self.yield_tokens(train_iter),
            specials=['<unk>', '<pad>']
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        logger.info(f"Vocabulary built with {len(self.vocab)} tokens")
        
    def text_pipeline(self, text: str) -> List[int]:
        tokens = self.tokenizer(text)
        return self.vocab(tokens)
    
    def label_pipeline(self, label: str) -> int:
        return int(label)
    
    def collate_batch(self, batch):
        label_list, text_list, lengths = [], [], []
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = self.text_pipeline(_text)
            # Truncate text to max_length
            processed_text = processed_text[:self.max_length]
            lengths.append(len(processed_text))
            text_list.append(processed_text)
            
        # Pad sequences to max_length
        padded_text = torch.zeros(len(text_list), self.max_length).long()
        for i, text in enumerate(text_list):
            padded_text[i, :lengths[i]] = torch.tensor(text)
            
        return torch.tensor(label_list), padded_text 