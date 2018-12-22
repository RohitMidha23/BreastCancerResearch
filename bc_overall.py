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



yoverall=y_predrfg+y_predrfe+y_predete+y_predetg+y_predsvm+y_predlr+y_prednb
for i in range(len(yoverall)):
	if yoverall[i]>3:
		yoverall[i]=1
	else:
		yoverall[i]=0
print(confusion_matrix(y_test, yoverall))  
#print(classification_report(y_test, y_pred))
print("overall ",accuracy_score(y_test, yoverall))



