from tensorflow.keras.layers import *

import tensorflow.keras as keras
import tensorflow as tf

def get_generator(input_shape, conditional=False):

    noise_input = Input(shape=(input_shape,))

    label_input = Input(shape=(10,)) if conditional else None

    model_input = concatenate([noise_input, label_input], axis=-1) if conditional else noise_input

    model = Dense(7*7*128, kernel_initializer=tf.initializers.HeNormal())(model_input)

    model = Reshape((7,7,128))(model)

    def block(x, num_filters):
        model = Conv2D(num_filters, (3,3), padding='same', kernel_initializer=tf.initializers.HeNormal())(x)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha=0.1)(model)
        model = UpSampling2D(size=(2,2), interpolation='bilinear')(model)
        return model

    model = block(model, 64) # --> [14,14,64]
    model = block(model, 32) # --> [28,28,32]
    model = Conv2D(1, 5, padding='same', activation='tanh')(model)

    return keras.Model(noise_input, model) if not conditional \
        else keras.Model([noise_input, label_input], model)


def get_discriminator(output_shape=1):

    image_input = Input(shape=(28,28,1))

    conv_params = {
        'padding': 'same',
        'kernel_initializer': tf.initializers.HeNormal()
    }

    def block(x, num_filters):
        model = Conv2D(num_filters, (5,5), strides=2, **conv_params)(x)
        model = Conv2D(num_filters, (3,3), **conv_params)(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha=0.1)(model)
        return model

    model = block(image_input, 32) # --> [14,14,32]
    model = block(model, 64) # --------> [ 7, 7,64]

    model = GlobalAveragePooling2D()(model)

    model = Dense(output_shape, activation=('sigmoid' if output_shape == 1 else 'softmax'))(model)

    return keras.Model(image_input, model)


def get_classifier():
    return get_discriminator(10)