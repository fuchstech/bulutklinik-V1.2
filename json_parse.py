import json
data = {"AGE":"2323","GENDER":"F","boy":"23","kilo":"23","meslek":"avukat","diger":"","evcil_hayvan":"1","hayvan_turu":"","ALLERGY":"2","alerji_turu":"kadircan","ALCOHOL_CONSUMING":"2","alkol_miktar":"yilda 363","SMOKING":"1","sigara_miktar":"","YELLOW_FINGERS":"0"}

def parse_js(data):
    js = json.dumps(data, indent=4)
    js = json.loads(js)
    #change str to integer
    for value in js:
        if value.isupper():
            if value == "GENDER" or value == "AGE":
                pass
            else:
                js[value] = int(js[value])
    #print(js)

    dataset = {'GENDER': 'F', 'AGE': 71, 'SMOKING': 1, 'YELLOW_FINGERS': 2, 'ANXIETY': 1, 'PEER_PRESSURE': 1, 'CHRONIC DISEASE': 2, 'FATIGUE': 2, 'ALLERGY': 1, 'WHEEZING': 2, 'ALCOHOL CONSUMING': 1, 'COUGHING': 2, 'SHORTNESS OF BREATH': 2, 'SWALLOWING DIFFICULTY': 1, 'CHEST PAIN': 1, 'LUNG_CANCER':'YES' }

    for values in dataset:
        for valuef in js:
            if valuef.isupper():
                if values == valuef:
                    #print(values,valuef)
                    dataset[values] = js[valuef]
    dataset = json.dumps(dataset,indent=4)   
    dataset=json.loads(dataset)
    print(dataset)
    return dataset
parse_js(data)
