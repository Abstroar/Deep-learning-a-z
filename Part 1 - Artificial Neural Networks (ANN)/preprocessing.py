import csv
from math import remainder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.src.legacy_tf_layers.core import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sympy.physics.units.systems.si import units
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.gen_nn_ops import Conv2D
from tensorflow.keras.layers import Dense





if __name__ == "__main__":


    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                               remainder='passthrough')  # applied to 1 and leave the others  bcz of passthrough
    X = np.array(ct.fit_transform(X))



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    print(le.classes_)
    print(le.transform(['Male']))
    print(sc.transform(ct.transform([[600,'France',1,40,3,60000,2,1,1,50000]])))
    ann = Sequential()
    ann.add(Dense(units=6, activation='relu', input_shape=(12,)))
    ann.add(Dense(units=6, activation='relu'))
    ann.add(Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    ann.summary()
    ann.fit(X_train, y_train, batch_size=32, epochs = 35)


    # OUT = ann.predict(sc.transform(ct.transform([[600,'France',1,40,3,60000,2,1,1,50000]])))
    # print(OUT > 0.5)

    y_pred = ann.predict(X_test)
    y_pred = (y_pred>0.5)
    confusion_matrix = confusion_matrix(y_test,y_pred)
    print(confusion_matrix)
    accuracy_score(y_test,y_pred)
    print(accuracy_score)