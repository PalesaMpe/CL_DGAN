import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class GANCheckpoint(Callback):

    def __init__(
        self,
        latent_dim,
        save_every=10,
        num_samples=500,
        checkpoint_dir="checkpoints"
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.save_every = save_every
        self.num_samples = num_samples
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.fixed_noise = tf.random.normal(
            shape=(num_samples, latent_dim),
            seed=42
        )

    def on_epoch_end(self, epoch, logs=None):

        current_epoch = epoch + 1

        if current_epoch % self.save_every != 0:
            return

        epoch_dir = os.path.join(
            self.checkpoint_dir,
            f"epoch_{current_epoch}"
        )

        os.makedirs(epoch_dir, exist_ok=True)

        # Save generator weights
        self.model.generator.save_weights(
            os.path.join(
                epoch_dir,
                "generator.weights.h5"
            )
        )

        # Save discriminator weights
        self.model.discriminator.save_weights(
            os.path.join(
                epoch_dir,
                "discriminator.weights.h5"
            )
        )

        generated_images = self.model.generator(
            self.fixed_noise,
            training=False
        )

        np.save(
            os.path.join(
                epoch_dir,
                "generated_images.npy"
            ),
            generated_images.numpy()
        )

        print(
            f"Saved checkpoint for epoch {current_epoch}"
        )