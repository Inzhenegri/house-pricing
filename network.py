import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sns.set_style(style='white')

rows = 10

path = 'housing.csv'
df = pd.read_csv(path)
# df = df.head(n=rows)
df = df.dropna()
df = df.drop(labels='ocean_proximity', axis=1)

plt.figure(figsize=(10, 8))

sns.heatmap(data=df.corr(), annot=True)

X = df.drop(labels='median_house_value', axis=1)
y = df.median_house_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X=X_train)
X_test = scaler.transform(X=X_test)


model = Sequential([
    Dense(units=32, input_shape=(8,), activation='relu'),
    Dense(units=1, activation='relu')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=X_train, y=y_train, epochs=1)
