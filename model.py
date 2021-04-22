import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas_profiling as pp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

# Read the Data Using Pandas
df=pd.read_csv('Poke.csv')
df.head()
# Dataset Summary
df.describe()
# Identify Outliers using Standard Deviation
missing_col = df[['Sp. Atk']]
mean = df[['Sp. Atk']].mean()
t = missing_col.fillna(mean)
u = round(t)
u.head()
df['Sp. Atk'] = u
# Delete the ‘Id’ column
del df['#']
df.isnull().sum()
df["Type 2"].fillna("Than Than Swe", inplace = True)
missing_col1 = df[['Total']]
mean = df[['Total']].mean()
d = missing_col1.fillna(mean)
v=round(d)
df['Total']=v
missing_col1 = df[['Sp. Atk']]
mean = df[['Sp. Atk']].mean()
d = missing_col1.fillna(mean)
v=round(d)
df['Sp. Atk']=v
#Using LTV&HTV
def outlier_detect(df):
    for i in df.describe().columns:
        Q1=df.describe().at['25%',i]
        Q3=df.describe().at['75%',i]
        IQR=Q3-Q1
        LTV=Q1-1.5*IQR
        UTV=Q3+1.5*IQR
        df[i]=df[i].mask(df[i]<LTV,LTV)
        df[i]=df[i].mask(df[i]>UTV,UTV)
    return df
out=outlier_detect(df)
scaler = StandardScaler()
print(scaler.fit(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]))
print(scaler.transform(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]))
# Separate the input & output from  Data into X and Y
X = df.iloc[:, [4,5,6,7,8,9]]
y = df.iloc[:, 11]
# Divide the data as train & test using train test split with test as 0.25 size
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=7)
model = LogisticRegression()
# Fit the train data in the model
model.fit(X_train, Y_train)
accuary_score = model.score(X_test, Y_test)
# Initialize parameters
num_folds = 2
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
# Initialize parameters
num_folds = 2
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
# Fitting the model and Extracting the results
results1 = cross_val_score(model, X, y, cv=kfold)
# Predict the output for test 
y_pred = model.predict(X_test)
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))
# Loading model to compare the results
model_pre = pickle.load(open('model.pkl','rb'))
print(model_pre.predict([[45,49,49,65,65,45]]))
