import requests

def finetune_model():
    API_URL = "https://1530-35-239-252-213.ngrok-free.app"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    response = requests.get(API_URL, headers)
    print(response.json())
    res = response.json()
    return res['model']