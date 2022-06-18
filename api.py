import requests as rq

url = "https://ephesus-api-3d2vvkkptq-ew.a.run.app/test"

text = "hello world"

params = {"sentence" : text}

response = rq.get(url, params=params)

if response.status_code == 200:
    response = response.json()
else:
    response = {}

name = response.get("name", "Toto")

print(name)
