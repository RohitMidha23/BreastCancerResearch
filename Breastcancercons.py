import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD, RMSprop

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history['val_loss'],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('binary_crossentropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True); p = ax.get_figure(); p.savefig('annplot.jpg')

dataset= pd.read_csv('/Users/rohit/BreastCancer/breast_cancer.csv',sep= ',')
del dataset['Unnamed: 32']
del dataset['id']


X = dataset.values[:,1:]
y = dataset.values[:,0]
u=[]

print ("Dataset Length: ", len(dataset))
print ("Dataset Shape:: ", dataset.shape)

for i in y:
	if(i=='M'):
		u.append(1)
	else:
		u.append(0)
y=np.array(u)
print(np.shape(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=23)

model = GaussianNB()
model.fit(X_train, y_train)
y_prednb = model.predict(X_test)
print(confusion_matrix(y_test,y_prednb))
#print(classification_report(y_test, y_pred))
print("Naive Bayes ",accuracy_score(y_test,y_prednb))
svc = SVC(kernel = 'linear',C=1, gamma=10, probability = True)
y_predsvm = svc.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test,y_predsvm ))
#print(classification_report(y_test, y_pred))
print("SVC ",accuracy_score(y_test,y_predsvm ))


clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
y_predlr = clf.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test,y_predlr))
#print(classification_report(y_test, y_pred))
print("Logistic regression ",accuracy_score(y_test,y_predlr))


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
dataset=dataset.drop(cols, axis=1)

cols = ['perimeter_mean',
        'perimeter_se',
        'area_mean',
        'area_se']
dataset=dataset.drop(cols, axis=1)

cols = ['concavity_mean',
        'concavity_se',
        'concave points_mean',
        'concave points_se']
dataset=dataset.drop(cols, axis=1)
X = dataset.drop('diagnosis',axis=1)
y = dataset['diagnosis']
u2=[]
for i in y:
	if(i=='M'):
		u2.append(1)
	else:
		u2.append(0)
y=np.array(u2)
print(np.shape(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=23)
rf=RandomForestClassifier(random_state=23,n_estimators = 200, max_features=6,criterion = 'entropy')
#rf=RandomForestClassifier(n_estimators=200,max_features=6,criterion="entropy")
rf.fit(X_train,y_train)
y_predrfe= rf.predict(X_test)
print(confusion_matrix(y_test,y_predrfe))
#print(classification_report(y_test, y_pred2))
print("Random forest entropy ",accuracy_score(y_test,y_predrfe))

rf=RandomForestClassifier(random_state=23,n_estimators=200,max_features=6,criterion="gini")
rf.fit(X_train,y_train)
y_predrfg = rf.predict(X_test)
print(confusion_matrix(y_test, y_predrfg))
#print(classification_report(y_test, y_pred))
print("Random forest gini ",accuracy_score(y_test,y_predrfg))

rf=ExtraTreesClassifier(random_state=23,n_estimators=200,max_features=6,criterion="entropy")
rf.fit(X_train,y_train)
y_predete = rf.predict(X_test)
print(confusion_matrix(y_test,y_predete))
#print(classification_report(y_test, y_pred))
print("Extra tree entropy ",accuracy_score(y_test,y_predete))

rf=ExtraTreesClassifier(random_state=23,n_estimators=200,max_features=6,criterion="gini")
rf.fit(X_train,y_train)
y_predetg = rf.predict(X_test)
print(confusion_matrix(y_test,y_predetg))
#print(classification_report(y_test, y_pred))
print("Extra tree gini ",accuracy_score(y_test,y_predetg))

#svc=SVC(kernel = 'linear', random_state = 0)


df = pd.read_csv('/Users/rohit/BreastCancer/breast_cancer.csv')
df = df.drop('Unnamed: 32', axis=1)

X = df.iloc[:, 2:].values
y = df['diagnosis'].values

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_te = train_test_split(X, y, test_size = 0.2, random_state = 23)

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
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=100, nb_epoch=512, validation_data = (X_test,y_te))
plot_loss_accuracy(history)

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_predann = []
for i in range(len(y_pred)):
        if y[i] == True:
                y_predann.append(1)
        else:
                y_predann.append(0)

print(confusion_matrix(y_te, y_pred))
a = accuracy_score(y_te, y_pred)
print(a)
yoverall=y_predrfg+y_predrfe+y_predete+y_predetg+y_predsvm+y_predlr+y_prednb+y_predann
for i in range(len(yoverall)):
	if yoverall[i]>3.5:
		yoverall[i]=1
	else:
		yoverall[i]=0

print(confusion_matrix(y_test, yoverall))
print(classification_report(y_test, yoverall))
print("overall ",accuracy_score(y_test, yoverall))
