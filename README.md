# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: s.monicka
RegisterNumber: 212221220033

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X = data[:, [0,1]]
y = data[:, 2]

print("Array value of X:")
X[:5]

print("Array value of Y:")
y[:5]

print("Exam 1-score graph:")
plt.figure()
plt.scatter(X[y==1][:, 0],X[y==1][:, 1], label="Admitted")
plt.scatter(X[y==0][:, 0],X[y==0][:, 1], label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print("Sigmoid function graph:")
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y,np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y)/ X.shape[0]
  return J,grad
  
print("X_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

print("Y_train_grad value:")
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J = -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  grad = np.dot(X.T,h-y) / X.shape[0]
  return grad 
  
print("Print res.x:")
X_train = np.hstack((np.ones((X.shape[0], 1)),X))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y), method='Newton-CG', jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max= X[:,0].min()-1, X[:,0].max()+1
  y_min, y_max= X[:,0].min()-1, X[:,0].max()+1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted") 
  plt.contour(xx, yy, y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,y)

print("Probability value:")
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob>=0.5).astype(int)
  
print("Prediction value of mean:")
np.mean(predict(res.x,X) == y)
*/
```

## Output:
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/bc7f9b57-32ec-49eb-8db4-371071b94ace)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/9fc96836-9437-4790-8a12-794082819656)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/354f2311-be92-4660-bf45-369259616309)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/e8e05bf1-317f-4281-86a4-87b5b061b01c)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/678c3905-7cfe-4b3e-8443-30a1f65bad47)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/10c7b727-98fa-4668-8093-c5aa470704f6)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/1f4fbb9c-d1ec-48c0-bae5-3042707df002)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/37660bad-2c3a-4b58-86b1-d50513ab8534)
![image](https://github.com/Monicka19/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/143497806/55edc781-16a3-4be6-95b4-e5e3450b1f6b)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

