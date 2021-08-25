from tensorflow.python.keras.callbacks import History
import matplotlib.pyplot as plt


def plot_loss(history: History):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)

    plt.show()
