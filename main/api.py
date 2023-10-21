import requests
import json

url = 'https://apihacuser.bulutklinik.com/'
payload = {
    "apiClientId": "9a4ba054-16ec-4d90-9ae0-5c340788e6cd",
    "apiSecretKey": "ROZT49Yr49bgs32UDEipLqs0Q1Bylqt6LL0qXmto",
    "apiUserName": "hackathon4@bulutklinik.com",
    "apiUserPassword": "bulutklinik2023.matiricie",
    "loginMode": "email"
}
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}

response = requests.request('POST', url, headers=headers, json=payload)
response.json()
