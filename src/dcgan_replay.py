import tensorflow as tf
from tensorflow.keras import metrics, models, losses


class DCGANReplay(models.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        replay_capacity=9000,
        replay_weight=0.5,
    ):
        super(DCGANReplay, self).__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.replay_weight = replay_weight
        self.replay_capacity = replay_capacity

        self.replay_buffer = tf.Variable(
            tf.zeros((replay_capacity, 64, 64, 1), dtype=tf.float32), trainable=False
        )

        self.replay_count = tf.Variable(0, dtype=tf.int32, trainable=False)

    def compile(self, d_optimizer, g_optimizer):
        super(DCGANReplay, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def add_images_to_replay(self, images):
        images = tf.cast(images, tf.float32)

        start = int(self.replay_count.numpy())

        number_to_add = int(images.shape[0])

        end = min(start + number_to_add, self.replay_capacity)

        number_that_fits = end - start

        if number_that_fits <= 0:
            print("Replay buffer is full.")
            return

        self.replay_buffer[start:end].assign(images[:number_that_fits])

        self.replay_count.assign(end)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            current_generated_images = self.generator(
                random_latent_vectors, training=True
            )

            real_predictions = self.discriminator(real_images, training=True)

            real_labels = tf.ones_like(real_predictions)
            real_noisy_labels = real_labels - 0.1 * tf.random.uniform(
                tf.shape(real_predictions)
            )

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)

            current_fake_predictions = self.discriminator(
                current_generated_images, training=True
            )

            current_fake_labels = tf.zeros_like(current_fake_predictions)

            current_fake_noisy_labels = current_fake_labels + 0.1 * tf.random.uniform(
                tf.shape(current_fake_predictions)
            )

            current_fake_loss = self.loss_fn(
                current_fake_noisy_labels, current_fake_predictions
            )
            replay_count = self.replay_count.read_value()

            def calculate_fake_loss_with_replay():
                replay_indices = tf.random.uniform(
                    shape=tf.stack([batch_size]),
                    minval=0,
                    maxval=replay_count,
                    dtype=tf.int32,
                )

                historical_images = tf.gather(self.replay_buffer, replay_indices)

                historical_predictions = self.discriminator(
                    historical_images, training=True
                )
                historical_labels = tf.zeros_like(historical_predictions)

                historical_noisy_labels = historical_labels + 0.1 * tf.random.uniform(
                    tf.shape(historical_predictions)
                )

                historical_fake_loss = self.loss_fn(
                    historical_noisy_labels, historical_predictions
                )

                combined_fake_loss = (
                    1.0 - self.replay_weight
                ) * current_fake_loss + self.replay_weight * historical_fake_loss

                return combined_fake_loss

            d_fake_loss = tf.cond(
                replay_count > 0,
                calculate_fake_loss_with_replay,
                lambda: current_fake_loss,
            )

            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, current_fake_predictions)
        gradients_of_discriminator = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )

        gradients_of_generator = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        self.d_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        self.g_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )

        self.d_loss_metric.update_state(d_loss)

        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}
