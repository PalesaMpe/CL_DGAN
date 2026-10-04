import numpy as np
from tensorflow.keras import layers, models

def create_discriminator():
    discriminator_input = layers.Input(shape=(64,64,1)) #takes an rgb image
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', use_bias=False)(discriminator_input)
    
    for filters in [128,256,512]:
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    
        x = layers.BatchNormalization(momentum=0.9)(x)
    
    x = layers.Flatten()(x)
    discriminator_output = layers.Dense(
        1,
        activation="sigmoid"
    )(x)
    return models.Model(discriminator_input, discriminator_output)