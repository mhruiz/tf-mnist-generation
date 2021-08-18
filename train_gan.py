from tqdm import tqdm
import tensorflow as tf
import numpy as np
import os

import dataset
import models
import utils

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

BATCH_SIZE = 32
NOISE_INPUT_LENGTH = 128

# Training dataset
tr_dataset = dataset.get_data(BATCH_SIZE)

# Validation dataset
# This is a random array which will be passed to our generator model
# Its predictions will be saved as images, so we will be able to check its quality
val_dataset = tf.random.normal(shape=(16, NOISE_INPUT_LENGTH))

# Define models
generator = models.get_generator(NOISE_INPUT_LENGTH)

discriminator = models.get_discriminator()


# Define loss functions

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(true_logits, fake_logits):

    real_loss = bce(tf.ones_like(true_logits), true_logits)
    fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)

    return real_loss + fake_loss

def generator_loss(fake_logits):
    return bce(tf.ones_like(fake_logits), fake_logits)

# Define optimizers
g_optimizer = tf.keras.optimizers.Adam()
d_optimizer = tf.keras.optimizers.Adam()


# Define training loop
@tf.function
def train_step(real_images, labels):

    noise_input_1 = tf.random.normal(shape=(BATCH_SIZE, NOISE_INPUT_LENGTH))
    noise_input_2 = tf.random.normal(shape=(BATCH_SIZE, NOISE_INPUT_LENGTH))

    with tf.GradientTape() as tp_gen, tf.GradientTape() as tp_dis:

        fake_images = generator(noise_input_1, training=True)

        true_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        d_loss = discriminator_loss(true_logits, fake_logits)

        fake_logits_2 = discriminator(generator(noise_input_2, training=True), training=True)

        g_loss = generator_loss(fake_logits_2)

    # Compute gradients
    g_gradients = tp_gen.gradient(g_loss, generator.trainable_variables)
    d_gradients = tp_dis.gradient(d_loss, discriminator.trainable_variables)

    # Apply gradients
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    return g_loss, d_loss


def train(num_epochs, epochs_between_val=10, path='training_progess_gan/'):

    if not os.path.exists(path):
        os.mkdir(path)

    num_batches = int(tf.data.experimental.cardinality(tr_dataset).numpy())

    for i in range(num_epochs):

        g_losses = np.zeros((num_batches,), dtype=np.float32)
        d_losses = np.zeros((num_batches,), dtype=np.float32)

        for b, (imgs, lbs) in tqdm(enumerate(tr_dataset), total=num_batches, ascii=True):
            g_loss, d_loss = train_step(imgs, lbs)

            g_losses[b] = g_loss
            d_losses[b] = d_loss

        g_losses = np.sum(g_losses) / num_batches
        d_losses = np.sum(d_losses) / num_batches

        print('Epoch', str(i+1).zfill(3), 'Gen. Loss:', g_losses, 'Disc. Loss:', d_losses)

        if i % epochs_between_val == 0:
            filename = path + 'epoch_' + str(i).zfill(3) + '.png'
            utils.generate_images(generator, val_dataset, filename)

    filename = path + 'epoch_' + str(num_epochs).zfill(3) + '.png'
    utils.generate_images(generator, val_dataset, filename)


train(20, epochs_between_val=2)