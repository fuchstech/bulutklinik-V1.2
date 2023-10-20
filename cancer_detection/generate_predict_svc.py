import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle


df = pd.read_csv("result_data.csv")
df.drop_duplicates(inplace=True)
print(df.shape)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['LUNG_CANCER']=encoder.fit_transform(df['LUNG_CANCER'])
df['GENDER']=encoder.fit_transform(df['GENDER'])
df.head()

con_col = ['AGE']
cat_col=[]
for i in df.columns:
    if i!='AGE':
        cat_col.append(i)

X=df.drop(['LUNG_CANCER'],axis=1)
y=df['LUNG_CANCER']

for i in X.columns[2:]:
    temp=[]
    for j in X[i]:
        temp.append(j-1)
    X[i]=temp





X_test = pd.read_csv("X_test.csv")
print(X_test.shape[0])
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X['AGE']=scaler.fit_transform(X_test[['AGE']])
print(X)

















with open('your_model_filename.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
    predictions = loaded_model.predict(X_test)

# Print the predictions
print("Predictions:", predictions)