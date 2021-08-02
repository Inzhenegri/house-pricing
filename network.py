import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from tabulate import tabulate


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

path = '/home/house_pricing/housing.csv'
df = pd.read_csv(path)
df = df.head(n=5)
df = df.dropna()
df = df.drop(labels='ocean_proximity', axis=1)

plt.figure(figsize=(8, 4))
sns.displot(data=df['median_house_value'])

print(tabulate(tabular_data=df, headers='keys', tablefmt='grid'))
