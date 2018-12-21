# BreastCancer

DecisionTreeRegressor gives the following Output : 
```python
[[72  3]
 [ 7 32]]
             precision    recall  f1-score   support

          0       0.91      0.96      0.94        75
          1       0.91      0.82      0.86        39

avg / total       0.91      0.91      0.91       114
```

RandomForestClassifier gives : 
```python 
[[74  1]
 [ 2 37]]
             precision    recall  f1-score   support

          B       0.97      0.99      0.98        75
          M       0.97      0.95      0.96        39

avg / total       0.97      0.97      0.97       114


```


XGBoost gives : 
```python
[[75  0]
 [ 2 37]]
             precision    recall  f1-score   support

          B       0.97      1.00      0.99        75
          M       1.00      0.95      0.97        39

avg / total       0.98      0.98      0.98       114
```


ANN gives : 
```python 
Epoch 512/512
512/512 [==============================] - 0s 18us/step - loss: 0.0028 - acc: 1.0000
[[35  0]
 [ 0 22]]
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        35
          1       1.00      1.00      1.00        22

avg / total       1.00      1.00      1.00        57

```
