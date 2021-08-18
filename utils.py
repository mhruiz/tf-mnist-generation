import matplotlib.pyplot as plt
import numpy as np
import math

def generate_images(model, noise, filename, labels=None):

    num_examples_per_axis = math.sqrt(noise.shape[0])

    assert num_examples_per_axis.is_integer(), 'Input noise must be a square number for plotting'

    generated_images = model(noise, training=False) if labels is None \
        else model([noise, labels], training=False)

    generated_images = generated_images.numpy()

    num_examples_per_axis = int(num_examples_per_axis)
    plt.subplots(num_examples_per_axis, num_examples_per_axis, figsize=(10,10))

    for i in range(noise.shape[0]):
        plt.subplot(num_examples_per_axis, num_examples_per_axis, i+1)

        image = (generated_images[i,...,0] + 1) / 2 # Convert to [0, 1]

        plt.imshow(image, cmap='gray')
        plt.axis('off')

        if labels is not None:
            plt.title(str(int(np.argmax(labels[i]))))

    plt.savefig(filename, bbox_inches='tight')
    plt.close()