import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 50
        
}


url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())

