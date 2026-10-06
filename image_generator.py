import numpy as np
import matplotlib.pyplot as plt

epochs = [10,50,100,180]
saved = {}

fig, axes = plt.subplots(
    len(epochs),
)

for epoch in epochs:
    saved[epoch] = np.load(
        f"checkpoints/epoch_{epoch}/generated_images.npy"
    )


for row in range(len(epochs)):
    epoch = epochs[row]
    images = saved[epoch].copy()

    # (-1, 1) to (0, 1)
    images = (images + 1.0) / 2.0
    images = images[:100]

    grid_rows = []
    for i in range(0, 100, 10):
        row_images = np.concatenate(
            images[i:i + 10],
            axis=1
        )

        grid_rows.append(row_images)

    grid = np.concatenate(
        grid_rows,
        axis=0
    )

    axes[row].imshow(
        grid,
        cmap="gray"
    )

    axes[row].axis("off")
    axes[row].set_title(f"G_{epoch}")

plt.tight_layout()
plt.show()