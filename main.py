from tensorflow.keras import metrics
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# from tb import tensorboard_callback
from functions import plot_loss


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sns.set_style(style='white')

rows = 5

path = 'housing.csv'
df = pd.read_csv(path)
# df = df.iloc[:rows]
df = df.dropna()
df = df.drop(labels='ocean_proximity', axis=1)

columns = df.columns

train_dataset = df.sample(frac=0.2)
test_dataset = df.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('median_house_value')
test_labels = test_features.pop('median_house_value')

normalizer = Normalization(axis=-1)
normalizer.adapt(data=np.array(train_features))

model = Sequential([
    normalizer,
    Dense(units=4, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.1), loss='mse', metrics=[metrics.RootMeanSquaredError(), 'mape'])

history = model.fit(
    x=train_features,
    y=train_labels,
    epochs=50,
    # validation_data=(test_features, test_labels)
)

model.save(filepath='my_model.h5')

losses = pd.DataFrame(model.history.history)

evaluated = model.evaluate(x=test_features, y=test_labels)

pyplot.plot(history.history['root_mean_squared_error'], label='root_mean_squared_error')
pyplot.plot(history.history['mae'], label='mae')
pyplot.plot(history.history['mape'], label='mape')

pyplot.legend()
pyplot.show()
