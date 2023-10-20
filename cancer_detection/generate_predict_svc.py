import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import pickle

class Generate():
    def __init__(self) -> None:
        
        df = pd.read_csv("result_data.csv")
        df.drop_duplicates(inplace=True)

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

        A=df.drop(['LUNG_CANCER'],axis=1)
        y=df['LUNG_CANCER']

        for i in A.columns[2:]:
            temp=[]
            for j in A[i]:
                temp.append(j-1)
            A[i]=temp



        #print(A['AGE'].values[0])
        age = A['AGE'].iloc[0]
        ages = pd.read_csv("X_test1.csv")["AGE"].values.tolist() #add2 bc range starts with 1 and an header column = 1
        fitted_ages = pd.read_csv("X_test.csv")["AGE"].values.tolist()
        if age in ages:
            age = fitted_ages[ages.index(age)]
            #print(age)
        else:
            for i in range(1,age*2):
                if age+i in ages:
                    #print(age, age + i ,ages.index(age+i))
                    #print(fitted_ages[ages.index(age+i)])
                    age = fitted_ages[ages.index(age+i)]
                    break
                if age-i in ages:
                    #print(age, age - i ,ages.index(age-i))
                    #print(fitted_ages[ages.index(age-i)])
                    age = fitted_ages[ages.index(age-i)]
                    break

        A['AGE'].iloc[0] = age
        self.A = A
    def make_predictions(self):
        with open('cancer_detection.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            
            predictions = loaded_model.predict(self.A)

        # Print the predictions
            pres =["NO","YES"]
            #print(predictions)
            #print("Predictions:", pres[predictions[0]])
            return pres[predictions[0]]

if __name__ == "__main__":
    print(Generate().make_predictions())