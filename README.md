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

6.Obtain the graph.  

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: v.sreeja
RegisterNumber:  212222230169

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
*/
```
## Output:
![logistic regression using gradient descent](sam.png)

![Screenshot (191)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/0bda247f-142b-430e-9c28-2c1089fcaba6)


![Screenshot (200)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/49499b4c-af17-4834-97f1-9011ff1bbe29)


![Screenshot (201)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/59fb1634-37cd-42d2-a868-c7330d838f31)



![Screenshot (194)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/25d64a73-c2a2-43ad-9252-ed345bcf75b9)

![Screenshot (195)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/2e0d17aa-c44a-4e5a-9a56-d5e301247e98)

![Screenshot (196)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/7290933e-1a1e-4289-bb13-cff1d07b8015)


![Screenshot (197)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/ef83d3ef-7fdb-4b99-99e5-fd9988ba7699)

![Screenshot (198)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/76abf075-e146-49c1-99bf-1e1a500e9996)


![Screenshot (199)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/bbf30107-3710-4a2e-97ba-1df58655b14f)


![Screenshot (203)](https://github.com/VelasiriSreeja/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118344328/5f1594c7-0dd0-46e2-9e33-349135e85ddc)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

