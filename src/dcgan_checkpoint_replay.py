import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class GANReplayCheckpoint(Callback):
    def __init__(
        self,
        latent_dim,
        save_every=10,
        num_samples=500,
        replay_samples=500,
        checkpoint_dir="checkpoints_replay",
    ):

        super().__init__()

        self.latent_dim = latent_dim
        self.save_every = save_every

        self.num_samples = num_samples
        self.replay_samples = replay_samples
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.fixed_noise = tf.random.normal(shape=(num_samples, latent_dim), seed=42)

    def on_epoch_end(self, epoch, logs=None):

        current_epoch = epoch + 1

        if current_epoch % self.save_every != 0:
            return

        epoch_dir = os.path.join(self.checkpoint_dir, f"epoch_{current_epoch}")

        os.makedirs(epoch_dir, exist_ok=True)

        self.model.generator.save_weights(
            os.path.join(epoch_dir, "generator.weights.h5")
        )
        self.model.discriminator.save_weights(
            os.path.join(epoch_dir, "discriminator.weights.h5")
        )

        evaluation_images = self.model.generator(self.fixed_noise, training=False)

        np.save(
            os.path.join(epoch_dir, "generated_images.npy"), evaluation_images.numpy()
        )

        replay_noise = tf.random.normal(shape=(self.replay_samples, self.latent_dim))
        replay_images = self.model.generator(replay_noise, training=False)
        self.model.add_images_to_replay(replay_images)

        print(f"Saved checkpoint for epoch {current_epoch}")
