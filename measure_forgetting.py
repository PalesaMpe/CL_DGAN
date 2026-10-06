import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from discriminator import create_discriminator


CHECKPOINT_DIR = "checkpoints"
epochs = [
    10,20,30,40,50,60,70,80,90, 100,110,120,130,140, 150,160,170, 180
]
"""
get all checkpoint samples
evaluate discriminator accuracy, how well does the discriminator predict generated samples as fake
evaluate bwt, retention over time
bwt = 1/k-1 sum(akj - ajj)
"""

def evaluate_fake_accuracy(discriminator, fake_images):
    predictions = discriminator.predict(
        fake_images,
        verbose=0
    ).reshape(-1)
    print("predictions", predictions)
    fake_accuracy = np.mean(
        predictions < 0.5
    )

    print("fake", fake_accuracy)
    return fake_accuracy

def measure_bwt(
        accuracy_matrix,
        checkpoint_index
):
    bwt_sum = 0.0
    if checkpoint_index == 0:
        return bwt_sum

    for j in range(checkpoint_index):
        bwt_sum += accuracy_matrix[checkpoint_index, j] - accuracy_matrix[j, j]
    return bwt_sum / checkpoint_index


discriminator = create_discriminator()

results = np.full(
    (len(epochs), len(epochs)),
    np.nan
)


for d_index, d_epoch in enumerate(epochs):
    discriminator_path = os.path.join(
        CHECKPOINT_DIR,
        f"epoch_{d_epoch}",
        "discriminator.weights.h5"
    )

    discriminator.load_weights(
        discriminator_path
    )

    for g_index, g_epoch in enumerate(epochs):
        if g_epoch > d_epoch:
            continue

        fake_path = os.path.join(
            CHECKPOINT_DIR,
            f"epoch_{g_epoch}",
            "generated_images.npy"
        )
        fake_images = np.load(
            fake_path
        ).astype("float32")
        accuracy = evaluate_fake_accuracy(
            discriminator,
            fake_images
        )
        results[d_index, g_index] = accuracy
        print(f"D{d_epoch}(G{g_epoch}) {accuracy}")

bwt_values = []

for k, epoch in enumerate(epochs):

    bwt = measure_bwt(
        results,
        k
    )

    bwt_values.append(
        bwt
    )


    print(f"D{epoch}: BWT = {bwt:.4f} ({bwt * 100:.2f}%)")

bwt_output = np.column_stack((epochs,np.array(bwt_values)))


np.savetxt(
    "bwt_by_epoch.csv",
    bwt_output,
    delimiter=",",
    header="epoch,bwt",
    comments="",
    fmt=["%d", "%.6f"]
)

plt.figure(figsize=(9, 5))
plt.plot(epochs[1:],bwt_values[1:],"o"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel(
    "Discriminator Checkpoint (Epoch)"
)

plt.ylabel(
    "Backward Transfer (BWT)"
)

plt.title(
    "Backward Transfer Across GAN Training"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    "bwt_over_training.png",
    dpi=300
)
plt.show()