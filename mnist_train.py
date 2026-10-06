from tensorflow.keras import datasets,utils,optimizers
import tensorflow as tf
from DGAN import DCGAN
import generator
import discriminator
from dcgan_checkpoint import GANCheckpoint
import numpy as np

# path  = kagglehub.dataset_download(
#     "gpiosenka/cards-image-datasetclassification"
# )
#
#
# print("Dataset downloaded to:", path)
# #tensorflow dataset
# train_data = utils.image_dataset_from_directory(path+'/train',
#                                                 labels=None,
#
#                                                 image_size=(64,64),
#                                                 batch_size=128,
#                                                 shuffle=True,
#                                                 seed=42,
#                                                 interpolation="bilinear")
(x_train, y_train), (_, _) = datasets.mnist.load_data()

print("Original dataset:", x_train.shape)

x_train = np.expand_dims(x_train, axis=-1)
train_data = tf.data.Dataset.from_tensor_slices(x_train)

def preprocess(image):
    #need to scale the original data -1 to 1
    image = tf.image.resize(image, (64, 64))
    image = (image - 127.5) / 127.5
    return image

#train = train_data.map(lambda x:preprocess(x))
train = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(60000, seed=42)
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

generator = generator.create_generator()
discriminator = discriminator.create_discriminator()

print("generator")
generator.summary()

print("discriminator")
discriminator.summary()

dcgan = DCGAN(
        discriminator=discriminator, generator=generator, latent_dim=100)

dcgan.compile(d_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
            g_optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
        )

checkpoint_callback = GANCheckpoint(
    latent_dim=100,
    save_every=10,
    num_samples=500,
    checkpoint_dir="checkpoints"
)


dcgan.fit(train, epochs=180,  callbacks=[checkpoint_callback]) #to train the model
