#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle    #used to save our trained model to the disk
import requests  #sends request to the server
import json      #print the result in our terminal

#importing the dataset
df1 = pd.read_csv(r"C:/Users/Monish Krishnan/Desktop/MAJOR PROJECT/cds2/Pkmeans3.csv")

#converting into different categories
df1["DayOT"] = df1["DayOT"].astype('category')
df1['DayOT'] = df1['DayOT'].cat.codes
dataframe = df1[['Beat', 'Arrest', 'Domestic', 'District', 'Crime_Date_Day', 'Crime_month', 'Year', 'DayOT', 'Labels']]

#train and test split of the data 
m = dataframe.shape[0]
n = 5 #features
k = 3 #classes

X = np.ones((m,n + 1))
y = np.array((m,1))

X[:,1] = dataframe['Beat'].values
X[:,2] = dataframe['Crime_Date_Day'].values
X[:,3] = dataframe['Crime_month'].values
X[:,4] = dataframe['Year'].values
X[:,5] = dataframe['DayOT'].values
#X[:,6] = dataframe['District'].values

#labels
y = df1['Labels'].values

#Mean normalization
for j in range(n):
    X[:, j] = (X[:, j] - X[:,j].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)

#saving model to disk
pickle.dump(dtree_model, open('model.pkl','wb'))

#loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))
model.predict([[0,2432,3,5,2016,0]])
