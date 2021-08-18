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

while True:
    val_labels = tf.random.uniform(shape=(16,), minval=0, maxval=10, dtype=tf.int32)
    if len(set(val_labels.numpy())) == 10:
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=10)
        break

# Define models
generator = models.get_generator(NOISE_INPUT_LENGTH, conditional=True)

discriminator = models.get_discriminator()

# Define and train a simple classifier model

classifier = models.get_classifier()

tr_dataset_classifier = dataset.get_data(BATCH_SIZE, shuffle=False)
num_batches = tf.data.experimental.cardinality(tr_dataset_classifier).numpy()

aux = int(np.floor(0.2 * num_batches))

val_dataset_classifier = tr_dataset_classifier.take(aux)
tr_dataset_classifier = tr_dataset_classifier.skip(aux)

tr_dataset_classifier = tr_dataset_classifier.shuffle(tf.data.experimental.cardinality(tr_dataset_classifier))

classifier.compile('adam', 'categorical_crossentropy', metrics=['acc'])

classifier.fit(tr_dataset_classifier, epochs=5, validation_data=val_dataset_classifier)



# Define loss functions

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

ce = tf.keras.losses.CategoricalCrossentropy()

def discriminator_loss(true_logits, fake_logits):

    real_loss = bce(tf.ones_like(true_logits), true_logits)
    fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)

    return real_loss + fake_loss

def generator_loss(fake_logits, real_label, pred_label):

    # Adversarial loss
    adv_loss = bce(tf.ones_like(fake_logits), fake_logits)

    # Content loss
    cnt_loss = ce(real_label, pred_label)

    return adv_loss + cnt_loss

# Define optimizers
g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


# Define training loop
@tf.function
def train_step(real_images, labels):
    
    noise_input_1 = tf.random.normal(shape=(labels.shape[0], NOISE_INPUT_LENGTH))
    noise_input_2 = tf.random.normal(shape=(labels.shape[0], NOISE_INPUT_LENGTH))

    with tf.GradientTape() as tp_gen, tf.GradientTape() as tp_dis:

        fake_images = generator([noise_input_1, labels], training=True)

        true_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        d_loss = discriminator_loss(true_logits, fake_logits)

        fake_images_2 = generator([noise_input_2, labels], training=True)

        fake_logits_2 = discriminator(fake_images_2, training=True)

        pred_labels = classifier(fake_images_2, training=True)

        g_loss = generator_loss(fake_logits_2, labels, pred_labels)

    # Compute gradients
    g_gradients = tp_gen.gradient(g_loss, generator.trainable_variables)
    d_gradients = tp_dis.gradient(d_loss, discriminator.trainable_variables)

    # Apply gradients
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

    return g_loss, d_loss


def train(num_epochs, epochs_between_val=10, path='training_progess_cgan/'):

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
            utils.generate_images(generator, val_dataset, filename, val_labels)

    filename = path + 'epoch_' + str(num_epochs).zfill(3) + '.png'
    utils.generate_images(generator, val_dataset, filename, val_labels)


train(50, epochs_between_val=2)