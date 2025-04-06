import torch
from torchtext.datasets import IMDB
from data.data_loader import IMDBDataProcessor
from models.lstm_model import LSTMClassifier
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model(model_path='model.pt', vocab_size=None):
    # Model parameters (same as in main.py)
    if vocab_size is None:
        logger.error("Vocabulary size must be provided to load_model")
        raise ValueError("vocab_size cannot be None")
        
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    
    # Initialize model
    model = LSTMClassifier(
        vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
        N_LAYERS, BIDIRECTIONAL, DROPOUT
    )
    
    # Load the trained model
    try:
        model.load_state_dict(torch.load(model_path))
        logger.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please train the model first.")
        raise
    except Exception as e:
        logger.error(f"Error loading model state_dict: {e}")
        raise
        
    model.eval()
    return model

def predict_sentiment(text, model, data_processor):
    if data_processor.vocab is None:
        logger.error("Data processor vocabulary is not built. Cannot predict.")
        return "Error: Vocabulary not built"
        
    # Process the input text
    try:
        tokens = data_processor.text_pipeline(text)
        if not tokens:
            logger.warning(f"Input text resulted in empty tokens: '{text}'")
            return "Neutral (empty input)"
            
        tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error processing text: {text}. Error: {e}")
        return "Error processing input"
    
    # Get prediction
    try:
        with torch.no_grad():
            prediction = model(tensor)
            # Output from log_softmax -> NLLLoss expects target 0 or 1
            # prediction.argmax will give 0 (neg) or 1 (pos)
            predicted_label = prediction.argmax(dim=1).item()
        
        # Map back to sentiment string
        return "Positive" if predicted_label == 1 else "Negative"
    except Exception as e:
        logger.error(f"Error during model prediction: {e}")
        return "Error during prediction"

def main():
    logger.info("Initializing data processor and building vocabulary...")
    # Initialize data processor
    data_processor = IMDBDataProcessor()
    
    # Load training data to build vocabulary
    try:
        # Re-instantiate iterator for vocab building
        vocab_build_iter = IMDB(split='train') 
        data_processor.build_vocabulary(vocab_build_iter)
        logger.info(f"Vocabulary built successfully with {len(data_processor.vocab)} words")
    except Exception as e:
        logger.error(f"Failed to load dataset or build vocabulary: {e}")
        return

    logger.info("Loading trained model...")
    # Load the model, passing the correct vocab size
    try:
        model = load_model(vocab_size=len(data_processor.vocab))
    except Exception:
        return # Error already logged in load_model
    
    # Sample reviews to test
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I really didn't enjoy this film. The story was confusing and the acting was poor.",
        "A masterpiece of modern cinema. The direction and cinematography were outstanding.",
        "Waste of time. The movie was boring and predictable.",
        "Great performances by the entire cast. The script was well-written and the pacing was perfect.",
        "It was okay, not great but not terrible either."
    ]
    
    print("\nTesting the model with sample reviews:")
    print("-" * 50)
    
    for review in sample_reviews:
        sentiment = predict_sentiment(review, model, data_processor)
        print(f"\nReview: {review}")
        print(f"Predicted Sentiment: {sentiment}")
        print("-" * 50)
    
    # Interactive testing
    print("\nNow you can enter your own reviews (type 'quit' to exit):")
    while True:
        try:
            review = input("\nEnter a review: ")
            if review.lower() == 'quit':
                break
            if not review.strip():
                print("Please enter some text.")
                continue
            sentiment = predict_sentiment(review, model, data_processor)
            print(f"Predicted Sentiment: {sentiment}")
        except EOFError:
            logger.info("EOF received, exiting interactive mode.")
            break
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, exiting.")
            break

if __name__ == "__main__":
    main() 