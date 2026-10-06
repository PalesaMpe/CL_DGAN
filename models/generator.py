from tensorflow.keras import layers, models

def create_generator():
#the generator acts as a decoder for the vae
        generator_input = layers.Input(shape=(100,))#Takes in noise, a 100 dimensional vector z that the genarator maps into an image
        x = layers.Reshape((1,1,100))(generator_input)
        x = layers.Conv2DTranspose(512, kernel_size=4, strides=1, padding='valid', use_bias=False)(x)

        for filters in [256, 128, 64]:
            x = layers.BatchNormalization(momentum=0.9)(x)
            x = layers.LeakyReLU(0.2)(x)
            x = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', use_bias=False)(x)

        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.LeakyReLU(0.2)(x)
        generator_output = layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', use_bias=False,activation='tanh')(x)

        return models.Model(generator_input, generator_output)

