# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 15:06:20 2023

@author: dhruv
"""

# Car Evaluation Database was derived from a simple hierarchical decision model 
# originally developed for the demonstration of DEX, M. Bohanec, V. Rajkovic: Expert system
# for decision making. Sistemica 1(1), pp. 145-157, 1990.). The model evaluates cars according 
# to the following concept structure:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# reading data
df = pd.read_csv("C:/Users/dhruv/.spyder-py3/Machine Learning sreeni/Random forest/Kaggle/car_evaluation.csv")
print(df.head())
print(df.describe())
# exloring data
print(df['class'].unique())
print(df['safety'].unique())
print(df['maintenance cost'].unique())
print(df['number of doors'].unique())

# describing the input and output variables
X = df.drop(['class'], axis = 1)
Y = df['class']

# converting into numeric data
import category_encoders as ce

encoder = ce.OrdinalEncoder(cols=['buying price', 'maintenance cost', 'number of doors', 'number of persons', 'lug_boot', 'safety'])
X = encoder.fit_transform(X)
print(X.head())

# test train split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=10)


# building the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, Y_train)

predicition = model.predict(X_test)

# measuring accuracy
from sklearn.metrics import accuracy_score
print('Accuracy = ', accuracy_score(Y_test, predicition))


















