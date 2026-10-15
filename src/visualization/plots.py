"""Shared figure style so every model's plots look the same in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_class_distribution(counts_by_split, class_names, path):
    """Grouped bars: images per class for train / val / test."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(class_names))
    n = len(counts_by_split)
    width = 0.8 / n
    for i, (split, counts) in enumerate(counts_by_split.items()):
        ax.bar(x + (i - (n - 1) / 2) * width, counts, width * 0.92,
               color=SERIES[i], label=split, zorder=2)
    ax.set_xticks(x, class_names)
    ax.set_ylabel("Images")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=n, loc="upper right")
    ax.set_title("Images per class")
    save(fig, path)


def _show(ax, img):
    """Show a 0-255 image that is either grayscale (H, W, 1) or RGB (H, W, 3)."""
    if img.shape[-1] == 1:
        ax.imshow(img[..., 0], cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(img.astype(np.uint8))


def plot_samples(x, y, class_names, path, per_class=6, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(per_class * 1.1, len(class_names) * 1.15))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(class_names):
        idx = np.flatnonzero(y == row)
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(chosen):
                _show(ax, x[chosen[col]])
        axes[row, 0].text(-0.15, 0.5, name, transform=axes[row, 0].transAxes,
                          ha="right", va="center", color=INK_2)
    fig.suptitle("Random training samples", x=0.02, ha="left", fontweight="bold", color=INK)
    save(fig, path)


def plot_mean_faces(x, y, class_names, path):
    fig, axes = plt.subplots(1, len(class_names), figsize=(len(class_names) * 1.4, 1.9))
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), class_names)):
        _show(ax, x[y == i].mean(axis=0))
        ax.set_title(name, fontsize=9, fontweight="normal", loc="center", color=INK_2)
        ax.axis("off")
    fig.suptitle("Mean face per class", x=0.02, ha="left", fontweight="bold", color=INK)
    save(fig, path)


def plot_history(history, path):
    """Loss and accuracy curves, train vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, metric, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[metric], color=SERIES[0], label="train")
        ax.plot(epochs, history[f"val_{metric}"], color=SERIES[1], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
    save(fig, path)

