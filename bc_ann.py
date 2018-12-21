import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/rohit/BreastCancer/breast_cancer.csv')
df = df.drop('Unnamed: 32', axis=1)

X = df.iloc[:, 2:].values
y = df['diagnosis'].values

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = Sequential()
model.add(Dense(output_dim=16, init='uniform', activation='relu', input_dim=30))
model.add(Dropout(p=0.1))
model.add(Dense(output_dim=16, init='uniform', activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=100, nb_epoch=512)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))