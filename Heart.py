import numpy as np
import pandas as pd
import pickle

df = pd.read_csv(r'D:\Python\Projects\Heart disease project\heart.csv')

print(df)

from sklearn.model_selection import train_test_split
X = df.drop(columns= 'target')
y = df['target']
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth= 4)
classifier.fit(X_train, y_train)

filename = 'heart-desease-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))