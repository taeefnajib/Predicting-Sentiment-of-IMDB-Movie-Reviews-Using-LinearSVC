# Import dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle
import os
from tqdm import tqdm 
from colorama import Fore, Back, Style 
import time

# Creating dataframe
df = pd.read_csv('data/data.csv')

# Defining feature variable and target variable
X = df['review']
y = df['sentiment']

# Train-test splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

for i in tqdm (range(6),  
               desc=Fore.GREEN + "Training",  
               ascii=False, ncols=75):
               # Feature extraction using TfidfVectorizer
               text_clf = Pipeline([('vectorizer', TfidfVectorizer()),('model', LinearSVC(random_state=0, loss="hinge", tol=0.01))])
               # Training the model
               text_clf.fit(X_train, y_train)
# Predicting the target variable on the test dataset
y_pred = text_clf.predict(X_test)
# Evaluation of the model
print("Accuracy:",round(accuracy_score(y_test, y_pred)*100, 2),"%")

for i in tqdm (range(6),  
               desc=Fore.GREEN + "Saving",  
               ascii=False, ncols=75):
               # Saving the model
               filename = 'model/model.sav'
               pickle.dump(text_clf, open(filename, 'wb'))
print("Model is saved at /model/model.sav")