import tensorflow as tf



log_dir = '/home/arseny/house-pricing/logs'
tf_writer = tf.summary.create_file_writer(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
