import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.layers import Flatten

def plot_roc(y_test, y_pred, model_name):
    fpr, tpr, thr = roc_curve(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=.5)  # roc curve for random model
    ax.grid(True)
    ax.set(title='ROC Curve for {} '.format(model_name),
           xlim=[-0.01, 1.01], ylim=[-0.01, 1.01])
    p = ax.get_figure()
    p.savefig('annplot.jpg')


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
model.add(Dense(16, init='uniform', activation='relu', input_dim=30))
model.add(Dropout(p=0.1))
model.add(Dense(16, init='uniform', activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, batch_size=100, nb_epoch=512)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
y_pred = (y_pred > 0.5)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))
plot_roc(y_test, y_pred_proba,'ANN')
