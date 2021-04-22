import requests
url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'hp':45,'attack':49, 'defense':49, 'sp_atk':65,'sp_def':65,'speed':45})
print(r.json())


