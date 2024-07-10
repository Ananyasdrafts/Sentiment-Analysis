import requests

data = {"text": "I think this movie is great"}
response = requests.post("http://localhost:5000/predict", json=data)

if response.status_code == 200:
    result = response.json()
    predicted_label = result['predicted_label']
    print(f"Predicted Label: {predicted_label}")
else:
    print(f"Error: {response.text}")
