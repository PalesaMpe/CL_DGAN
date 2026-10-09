import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, optimizers
from dcgan_replay import DCGANReplay
from models import generator, discriminator
from dcgan_checkpoint_replay import GANReplayCheckpoint

(x_train,y_train ),(_,_)= (datasets.mnist.load_data())

x_train = np.expand_dims(
    x_train,
    axis=-1
)


def preprocess(image):
    image = tf.image.resize(image,(64, 64))
    image = tf.cast(image,tf.float32)
    image = (image - 127.5) / 127.5
    return image

train = (tf.data.Dataset.from_tensor_slices(x_train).shuffle(
        60000,42).map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(32)
    .prefetch(
        tf.data.AUTOTUNE
    )
)

generator = generator.create_generator()
discriminator = discriminator.create_discriminator()

dcgan = DCGANReplay(
    discriminator=discriminator,generator=generator, latent_dim=100,
    #18 checkpoints × 500 samples
    replay_capacity=9000,
    replay_weight=0.5
)

dcgan.compile(d_optimizer=optimizers.Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.999),
            g_optimizer=optimizers.Adam(learning_rate=0.0002,beta_1=0.5,beta_2=0.999)
    )

checkpoint_callback = GANReplayCheckpoint(
    latent_dim=100,
    save_every=10,
    num_samples=500,
    replay_samples=500,
    checkpoint_dir="checkpoints_replay_mnist"
)

dcgan.fit(train,epochs=180,callbacks=[checkpoint_callback])