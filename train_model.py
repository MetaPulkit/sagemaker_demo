# save_model.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def save_model(model_name: str, save_directory: str):
    # Load pretrained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save model and tokenizer locally
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    print(f"Model saved to {save_directory}")

# Example usage
if __name__ == "__main__":
    model_name = 'bert-base-uncased'  # Use any other pretrained model
    save_directory = './model'
    save_model(model_name, save_directory)
