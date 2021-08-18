import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import data

def get_data(batch_size, shuffle=True):

    # Download datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Join all images, we will use all of them to train our generator
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # Reshape to (N, h, w, c)
    x = x.reshape(*(list(x.shape) + [1])).astype(np.float32)

    # Normalize to [-1, 1]
    x = (x - 127.5) / 127.5

    # Convert labels to 'one_hot' format
    y = tf.keras.utils.to_categorical(y.astype(np.float32), num_classes=10)
    
    # Convert to TF Dataset
    dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(x.shape[0])

    dataset = dataset.batch(batch_size)

    return dataset