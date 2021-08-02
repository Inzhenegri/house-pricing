import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from tabulate import tabulate


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

rows = 1000

path = 'housing.csv'
df = pd.read_csv(path)
df = df.head(n=rows)
df = df.dropna()
df = df.drop(labels='ocean_proximity', axis=1)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=df.median_house_value,
    y=df.total_rooms,
    hue=df.population,
    palette=sns.color_palette(palette='coolwarm', as_cmap=True)
)
sns.displot(x=df.median_house_value, kind='kde')

# print(tabulate(tabular_data=df, headers='keys', tablefmt='grid'))

plt.show()
