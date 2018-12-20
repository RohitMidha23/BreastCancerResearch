import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df= pd.read_csv('/Users/rohit/Desktop/bc/breast_cancer.csv',sep= ',')
df = df.drop('Unnamed: 32', axis=1)

cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)

cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)

cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

X = df.drop('diagnosis',axis=1)
y = df['diagnosis']

# to one hot encode y 
data = y
values = array(data)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=23)

model = DecisionTreeRegressor(random_state=23)

model.fit(X_train,y_train)

predicted = model.predict(X_test)



print(confusion_matrix(y_test.argmax(axis=1), predicted.argmax(axis=1)))  
print(classification_report(y_test,predicted))
