import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

from tb import tensorboard_callback


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sns.set_style(style='white')

rows = 5

path = 'housing.csv'
df = pd.read_csv(path)
# df = df.iloc[:rows]
df = df.dropna()
df = df.drop(labels='ocean_proximity', axis=1)

plt.show()

columns = df.columns

plt.figure(figsize=(10, 8))
sns.heatmap(data=df.corr(), annot=True)


scaler = MinMaxScaler()

df = scaler.fit_transform(df)
df = pd.DataFrame(df, columns=columns)

X = df.drop(labels='median_house_value', axis=1)
y = df.median_house_value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = Sequential([
    Dense(units=32, input_shape=(8,), activation='relu'),
    Dense(units=10, activation='relu'),
    Dense(units=1, activation='relu')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mape', metrics=['mape'])

history = model.fit(
    x=X_train,
    y=y_train,
    epochs=50,
    # callbacks=[tensorboard_callback]
)

model.save(filepath='my_model.h5')

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()
