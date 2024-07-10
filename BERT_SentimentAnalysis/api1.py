from flask import Flask, request, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the fine-tuned weights of your sentiment analysis model
model.load_state_dict(torch.load("bert-movie-review.model", map_location = torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        input_text = request.form['text']

        # Tokenize and process the input text
        inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return render_template('index.html', sentiment_label=predicted_class)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
