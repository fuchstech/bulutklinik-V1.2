import json

j = open("survey lung cancer.json")
data = json.load(j)
for i in data:
    print(i)