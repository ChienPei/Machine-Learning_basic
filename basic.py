#!/usr/bin/env python
# coding: utf-8

# import packages
# Note: You cannot import any other packages!
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import random

# Global attributes
# Do not change anything here except TODO 1 
StudentID = 'StudentID' # TODO 1 : Fill your student ID here
input_dataroot = 'input.csv' # Please name your input csv file as 'input.csv'
output_dataroot = StudentID + '_basic_prediction.csv' # Output file will be named as '[StudentID]_basic_prediction.csv'

input_datalist =  [] # Initial datalist, saved as numpy array
output_datalist =  [] # Your prediction, should be 20 * 2 matrix and saved as numpy array
                      # The format of each row should be [Date, TSMC_Price_Prediction] 
                      # e.g. ['2021/10/15', 512]

# You can add your own global attributes here

# Read input csv to datalist
with open(input_dataroot, newline='') as csvfile:
    input_datalist = np.array(list(csv.reader(csvfile)))

def SplitData():
    # TODO 2: Split data, 2021/10/15 ~ 2021/11/11 for testing data, and the other for training data and validation data 
     index_split = 0 
     for i in range(0,len(input_datalist.T[0])):
        # print(input_datalist.T[2][i])
        if input_datalist.T[2][i] == '0' :
            index_split = i
            break 
     training_data = input_datalist[0:index_split]
     testing_data= input_datalist[index_split:len(input_datalist.T[0])]
     return testing_data,training_data

# process training data
def PreprocessData(training_data,testing_data):
    # TODO 3: Preprocess your data  e.g. split datalist to x_datalist and y_datalist
    #x_datalist_MTK = training_data.T[1]
    #y_datalist_TSMC = training_data.T[2]

    x = training_data[:, 1]
    y = training_data[:, 2]
    t_x = testing_data[:, 1]
    t_y = testing_data[:, 2]

    x = x.reshape(x.shape[0],1)
    y = y.reshape(y.shape[0],1)
    t_x = t_x.reshape(t_x.shape[0],1)
    t_y = t_y.reshape(t_y.shape[0],1)
  
    x = x.astype(float)
    y = y.astype(float)
    t_x = t_x.astype(float)
    t_y = t_y.astype(float)
    
    return x,y,t_x,t_y

def Regression(X, y, theta, alpha, num_iters):
    # TODO 4: Implement regression
    m = np.size(y)
    n = np.size(theta)
    tmp = np.zeros((n,1)) #2*1
    X = X.astype(float)
    y = y.astype(float)
    
    for iter in range(1, num_iters+1 ):
        for initer in range(0, n):
            tmp[initer] = theta[initer] - alpha * (1.0 / m) * sum(np.dot((np.dot(X , theta) - y).T , X[:,initer]))
        theta = tmp
        
    return theta

def CountLoss(y,t_y):
    # TODO 5: Count loss of training and validation data
    loss = 0 
    for i in range(len(t_y)):
        a = (y[i]-t_y[i])/(y[i])
        if a>= 0:
            loss += ((y[i]-t_y[i])/(y[i]))
        else:
            loss -= ((y[i]-t_y[i])/(y[i]))
    loss/=len(y)

    return loss

def MakePrediction(theta,t_x,t_y):
    # TODO 6: Make prediction of testing data 
    y_hat = t_y
    for i in range(0, len(t_x) ): 
        
        # print('x  ',x[i])
        # print('y  ' , y[i])
        y_hat[i]= t_x[i]*theta[1]+theta[0]
        # print(' t_y[i]  ', t_y[i])
        # print('----')
    
    return y_hat  

# TODO 7: Call functions of TODO 2 to TODO 6, train the model and make prediction
testing_data,training_data = SplitData()
x,y,t_x,t_y = PreprocessData(training_data,testing_data)

iterations = 300
alpha = 0.0000005

oneX = np.concatenate((np.ones((np.size(y),1)), x), axis = 1) # 加1
theta = np.zeros((2,1)) 
theta = Regression(oneX, y, theta, alpha, iterations)


predic = MakePrediction(theta,t_x,t_y)
output_datalist= predic

x,y,t_x,t_y = PreprocessData(training_data,testing_data)
mape = CountLoss(y,predic)

# Write prediction to output csv
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Date', 'TSMC Price'])
    date = [['2021/10/15'],['2021/10/18'],['2021/10/19'],['2021/10/20'],['2021/10/21'],['2021/10/22'],['2021/10/23'],['2021/10/26'],['2021/10/27'],['2021/10/28'],['2021/10/29']
    ,['2021/11/1'],['2021/11/2'],['2021/11/3'],['2021/11/4'],['2021/11/5'],['2021/11/8'],['2021/11/9'],['2021/11/10'],['2021/11/11']]
    output_datalist = np.concatenate((date, output_datalist), axis = 1)
    for row in output_datalist:
        writer.writerow(row)