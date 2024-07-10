import requests

data = {"text": "I think this movie is great"}
response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
