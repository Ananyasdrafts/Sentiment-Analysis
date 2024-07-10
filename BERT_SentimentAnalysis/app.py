from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

app = Flask(__name__)

# Define the path to your local model directory
model_dir = "bert_sentiment_model"

# Check if the model directory exists
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"The directory '{model_dir}' does not exist. Please ensure the model is saved in this directory.")

# Load the BERT model and tokenizer from the local directory
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

# Define sentiment labels mapping
sentiment_labels = {
    0: "negative",
    1: "somewhat negative",
    2: "neutral",
    3: "somewhat positive",
    4: "positive",
}

# Function to classify sentiment
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = sentiment_labels[predicted_class]
    return sentiment

@app.route('/classify_sentiment', methods=['POST'])
def classify_sentiment_endpoint():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "Please provide 'text' in the request body"}), 400

    sentiment = classify_sentiment(text)
    response_data = {
        "text": text,
        "sentiment": sentiment,
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
