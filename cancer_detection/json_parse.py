import json
import pandas as pd
data = {"AGE":"2323","GENDER":"F","boy":"23","kilo":"23","meslek":"avukat","diger":"","evcil_hayvan":"1","hayvan_turu":"","ALLERGY":"2","alerji_turu":"kadircan","ALCOHOL_CONSUMING":"2","alkol_miktar":"yilda 363","SMOKING":"1","sigara_miktar":"","YELLOW_FINGERS":"0"}

data2={
    "complaints": [
        "'COUGHING': 2,",
        "ates",
        "ANXIETY"
    ],
    "additionalOptions": {
        "COUGHING-dropdown": "dropdown",
        "SHORTNESS_OF_BREATH-dropdown": "dropdown",
        "WHEEZING-dropdown": "dropdown",
        "CHEST_PAIN-checkboxes": "checkboxes",
        "bayilma-checkboxes": "checkboxes",
        "kilo-kaybi-checkboxes": "checkboxes",
        "ates-dropdown": "dropdown",
        "ANXIETY-dropdown": "dropdown",
        "SWALLOWING_DIFFICULTY-dropdown": "dropdown",
        "bas-checkboxes": "checkboxes",
        "karin-checkboxes": "checkboxes",
        "karinAgrisi-checkboxes": "checkboxes"
    },
    "sure": "66",
    "bayilmasayisi": ""
}
def parse_js_form(data2):
    data2 = json.dumps(data2,indent=4)
    data2 = json.loads(data2)
    print(json.loads(json.dumps(data2["complaints"])))
    
def parse_js_kayit(data,type="dict"):
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

    dataset = {'GENDER': 'F', 'AGE': 71, 'SMOKING': 1, 'YELLOW_FINGERS': 2, 'ANXIETY': 1, 'PEER_PRESSURE': 1, 'CHRONIC DISEASE': 2, 'FATIGUE ': 2, 'ALLERGY ': 1, 'WHEEZING': 2, 'ALCOHOL CONSUMING': 1, 'COUGHING': 2, 'SHORTNESS OF BREATH': 2, 'SWALLOWING DIFFICULTY': 1, 'CHEST PAIN': 1, 'LUNG_CANCER':'1' }

    for values in dataset:
        for valuef in js:
            if valuef.isupper():
                if values == valuef:
                    #print(values,valuef)
                    dataset[values] = js[valuef]
    datasetj = json.dumps(dataset,indent=4)   
    datasetj=json.loads(datasetj)
    print(datasetj,"\n",dataset)
    if type == "json":
        return datasetj
    if type == "dict":
        return dataset

def save_csv(dataset):
    result_data = pd.read_csv("result_data.csv")
    df = pd.DataFrame(dataset,index=[0])
    df = pd.concat([result_data,df])
    df.to_csv("result_data.csv",index=0)
#parse_js_form(data2)
dataset = parse_js_kayit(data,"dict")
save_csv(dataset)
