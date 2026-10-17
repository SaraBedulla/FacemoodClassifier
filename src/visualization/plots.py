"""Shared figure style so every model's plots look the same in the report."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

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


def plot_history(history, path, fine_tune_start=None):
    """Loss and accuracy curves, train vs validation.

    fine_tune_start: number of head-only epochs before fine-tuning began (transfer models);
    a dashed line marks where fine-tuning starts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for ax, metric, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(epochs, history[metric], color=SERIES[0], label="train")
        ax.plot(epochs, history[f"val_{metric}"], color=SERIES[1], label="validation")
        if fine_tune_start:
            ax.axvline(fine_tune_start + 0.5, color=MUTED, linestyle="--", linewidth=1,
                       label="fine-tuning starts")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend()
    save(fig, path)


def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix"):
    """Row-normalised confusion matrix: each row shows how one true class was predicted."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE, SERIES[0]])
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = 0.75 * len(class_names) + 1.5
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{rates[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color=SURFACE if rates[i, j] > 0.55 else INK_2)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks, class_names, rotation=45, ha="right")
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    save(fig, path)


def plot_predictions(x, y_true, y_pred, class_names, path, per_group=8, seed=0):
    """Two rows of test images: correctly and incorrectly classified, labelled true / predicted."""
    rng = np.random.default_rng(seed)
    groups = [("Correct", np.flatnonzero(y_true == y_pred)),
              ("Incorrect", np.flatnonzero(y_true != y_pred))]
    fig, axes = plt.subplots(2, per_group, figsize=(per_group * 1.3, 3.4))
    for row, (name, idx) in enumerate(groups):
        chosen = rng.choice(idx, size=min(per_group, len(idx)), replace=False)
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col < len(chosen):
                i = chosen[col]
                _show(ax, x[i])
                ax.set_title(f"{class_names[y_true[i]]}\n→ {class_names[y_pred[i]]}", fontsize=7,
                             fontweight="normal", loc="center",
                             color=INK_2 if row == 0 else SERIES[1])
        axes[row, 0].text(-0.15, 0.5, name, transform=axes[row, 0].transAxes,
                          ha="right", va="center", color=INK_2)
    fig.suptitle("Test predictions (true → predicted)", x=0.02, ha="left",
                 fontweight="bold", color=INK)
    save(fig, path)


def plot_data_size(table, path, metric="test_macro_f1", title="Macro-F1 vs training-set size"):
    """One line per model: mean metric (± std over seeds) against the fraction of training data.

    table needs columns: model, train_fraction, <metric>_mean, <metric>_std.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for color, (model, rows) in zip(SERIES, table.groupby("model", sort=False)):
        rows = rows.sort_values("train_fraction")
        ax.errorbar(rows["train_fraction"] * 100, rows[f"{metric}_mean"],
                    yerr=rows[f"{metric}_std"].fillna(0), color=color, marker="o",
                    capsize=3, label=model)
    ax.set_xscale("log")
    fractions = sorted(table["train_fraction"].unique() * 100)
    ax.set_xticks(fractions, [f"{f:g}%" for f in fractions])
    ax.minorticks_off()
    ax.set_xlabel("Training data used")
    ax.set_ylabel(metric.replace("test_", "").replace("_", "-"))
    ax.set_title(title)
    ax.legend()
    save(fig, path)



def plot_model_comparison(table, path, metric="test_macro_f1", title="Test macro-F1 per model"):
    """Grouped bars: one group per (dataset, labels) setting, one bar per model.

    table needs columns: setting, model, <metric>_mean.
    """
    settings = list(dict.fromkeys(table["setting"]))
    models = list(dict.fromkeys(table["model"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(settings))
    width = 0.8 / len(models)
    for i, (color, model) in enumerate(zip(SERIES, models)):
        rows = table[table["model"] == model].set_index("setting").reindex(settings)
        values = rows[f"{metric}_mean"].to_numpy()
        bars = ax.bar(x + (i - (len(models) - 1) / 2) * width, values, width * 0.92,
                      color=color, label=model, zorder=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, settings)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("test_", "").replace("_", "-"))
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(models), loc="upper left")
    ax.set_title(title)
    save(fig, path)


def plot_fine_tuning_gain(table, path):
    """Validation accuracy of each transfer run with the frozen backbone vs after fine-tuning.

    table needs columns: setting, model, frozen_val_accuracy, best_val_accuracy.
    """
    labels = [f"{m}\n{s.replace(' ', chr(10))}" for m, s in zip(table["model"], table["setting"])]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(table))
    for offset, column, color, name in [(-0.2, "frozen_val_accuracy", BASELINE, "frozen backbone (head only)"),
                                        (0.2, "best_val_accuracy", SERIES[0], "after fine-tuning")]:
        bars = ax.bar(x + offset, table[column], 0.37, color=color, label=name, zorder=2)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Effect of fine-tuning the pretrained backbone")
    save(fig, path)
