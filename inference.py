# inference.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import json

def model_fn(model_dir):
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer(input_data['inputs'], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).tolist()
    return {"predictions": predictions}

def input_fn(request_body, content_type='application/json'):
    # Parse the input data
    if content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError("Unsupported content type")

def output_fn(prediction, accept='application/json'):
    # Return predictions as JSON
    return json.dumps(prediction), accept
