# FacemoodClassifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaraBedulla/FacemoodClassifier/blob/main/notebooks/FacemoodClassifier_colab.ipynb)

Depression-related facial expression classification with deep learning.

This project classifies facial-expression images into emotions and into depression-related
expression patterns. It compares a **CNN trained from scratch** with **pretrained transfer-learning
models (MobileNetV2 and ResNet50)**.

## Research question

> How does a CNN trained from scratch compare with transfer learning (MobileNetV2, ResNet50)
> for facial expression classification, in terms of accuracy, efficiency, and the amount of
> training data needed?

## Datasets

Both datasets use the same 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.
Neither is included in this repository; download them and unzip into `data/raw/`.
Full description, links and class counts: [DATASET.md](DATASET.md).

| | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (default) | [RAF-DB](http://www.whdeng.cn/RAF/model1.html) |
|---|---|---|
| Images | 35,887 | 15,339 |
| Format | 48×48 grayscale | 100×100 colour, aligned |
| Labels | 1 annotator, known to be noisy | ~40 annotators per image |
| Split (train / val / test) | 25,838 / 2,871 / 7,178 | 11,043 / 1,228 / 3,068 |
| Largest class | happy 25.0% | happy 38.8% |
| Smallest class | disgust 1.5% | fear 2.3% |
| Folder | `data/raw/fer2013/` with `train/<emotion>/`, `test/<emotion>/` | `data/raw/rafdb/` with `train/<1-7>/`, `test/<1-7>/` ([Kaggle copy](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)) or the official `EmoLabel/` + `Image/aligned/` layout |

The validation set is a stratified 10% of the original training set. FER-2013's CSV version
(`fer2013.csv`) is also supported.

### Label schemes

| Scheme | Classes | Purpose |
|---|---|---|
| `emotion7` | angry, disgust, fear, happy, sad, surprise, neutral | Standard emotion recognition |
| `affect3` | **negative** (angry, disgust, fear, sad), **neutral**, **positive** (happy, surprise) | Depression-related expression patterns |

The `affect3` grouping reflects findings that depression is associated with more negative and
less positive facial affect.

## Models

| Model | Input | Description |
|---|---|---|
| Scratch CNN | native (48×48×1 FER-2013, 100×100×3 RAF-DB) | 4 convolutional blocks (Conv–BN–ReLU ×2, max-pool, dropout), global average pooling, dense head |
| MobileNetV2 | 224×224×3 | ImageNet-pretrained backbone (2.3M parameters), new head; fine-tunes the last 3 blocks (`block_13` onwards) |
| ResNet50 | 224×224×3 | ImageNet-pretrained backbone (23.6M parameters), new head; fine-tunes the last stage (`conv5`) |

The transfer models resize each face to 224×224 and copy grayscale images to 3 channels. Each model
applies its own ImageNet preprocessing inside the network, so every model takes raw 0–255 pixels.
The new head is global average pooling → dropout (0.3) → softmax. They train in two phases:

1. **Head**: the backbone is frozen and only the new head trains (Adam, learning rate 1e-3, up to 10 epochs).
2. **Fine-tune**: the top backbone stage is unfrozen and trains together with the head (learning rate 1e-5)
   until early stopping. BatchNorm layers stay frozen so the ImageNet statistics are kept.

All models use the same data splits, augmentation (flip, rotation, zoom, shift), class weighting,
and evaluation, so the comparison is fair.

## Experiments

All three models were trained on both datasets with both label schemes (12 runs, seed 42), with the
same default settings. The one exception is the RAF-DB `emotion7` scratch CNN, which used
`--lr 0.0003 --epochs 100 --patience 20`.

**Metrics:** accuracy, macro-F1 (the classes are imbalanced), per-class recall, confusion matrix,
number of parameters, training time, and inference time per image.

The code also supports repeating runs with several seeds (`--seed`) and training on part of the data
(`--train-fraction`); `compare.py` then reports mean ± std and plots macro-F1 against training-set size.
These extra experiments were planned but not run in the available GPU time.

## Results

Test accuracy / macro-F1 (one run per setting):

| Setting | Scratch CNN | MobileNetV2 | ResNet50 |
|---|---|---|---|
| FER-2013, `emotion7` | **64.8% / 0.633** | 63.0% / 0.605 | 63.8% / 0.603 |
| FER-2013, `affect3` | 74.5% / 0.723 | 76.0% / 0.733 | **76.3% / 0.740** |
| RAF-DB, `emotion7` | **78.7% / 0.700** | 65.4% / 0.575 | 69.7% / 0.608 |
| RAF-DB, `affect3` | **82.4% / 0.808** | 75.6% / 0.740 | 78.1% / 0.762 |

The CNN trained from scratch had the best macro-F1 in three of four settings and was the fastest at
inference. Fine-tuning was essential for the pretrained models: with a frozen ImageNet backbone they
reached only 45–67% validation accuracy, and fine-tuning the top stage added 10–19 points. The full
analysis is in the project report.

## Project structure

```
FacemoodClassifier/
├── notebooks/
│   └── FacemoodClassifier_colab.ipynb   # full pipeline, runs on Colab or locally
├── src/
│   ├── config/
│   │   └── config.py            # paths and constants
│   ├── data/
│   │   ├── labels.py            # emotion7 / affect3 label schemes
│   │   ├── prepare.py           # reads raw RAF-DB / FER-2013, builds fixed splits
│   │   ├── dataset.py           # loading splits, class weights, tf.data + augmentation
│   │   └── eda.py               # exploratory data analysis
│   ├── models/
│   │   ├── registry.py          # models available for training, input size, layers to fine-tune
│   │   ├── scratch_cnn.py       # CNN from scratch
│   │   └── transfer.py          # MobileNetV2 / ResNet50 with a new head, unfreezing for fine-tuning
│   ├── training/
│   │   └── train.py             # trains one model configuration (two phases for transfer models)
│   ├── evaluation/
│   │   ├── evaluate.py          # test-set metrics, confusion matrix, sample predictions per run
│   │   └── compare.py           # comparison tables and figures (mean ± std over seeds)
│   └── visualization/
│       └── plots.py             # shared figure style
├── DATASET.md                   # dataset links, description and splits
├── requirements.txt
└── README.md
```

Created when you run the code (not committed):

```
data/
├── raw/                         # datasets as downloaded (e.g. raw/rafdb/)
└── processed/                   # fixed train/val/test splits
results/
├── figures/                     # EDA and comparison figures
├── tables/                      # EDA class counts and comparison tables
└── runs/                        # one folder per training run
```

## Setup

Requires Python 3.9–3.12 (TensorFlow 2.16 has no wheels for newer versions).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tensorflow-metal` (Apple Silicon GPU) is only installed on macOS; pip skips it elsewhere.
TensorFlow has no GPU support on native Windows, so a local Windows setup runs on the CPU. That is fine
for the scratch CNN and quick tests, but train the transfer models on Colab (below).

## Running on Google Colab

Open [`notebooks/FacemoodClassifier_colab.ipynb`](notebooks/FacemoodClassifier_colab.ipynb) with the badge
above, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The notebook clones this
repository, downloads FER-2013 from Kaggle, and runs the same commands as below. The same notebook also runs
locally, where it uses the dataset already in `data/raw/`.

## Usage

```bash
# 1. Build the fixed train/val/test splits (run once)
python -m src.data.prepare --dataset fer2013 --source data/raw/fer2013
python -m src.data.prepare --dataset rafdb --source data/raw/rafdb

# 2. Explore the data
python -m src.data.eda

# 3. Train a model
python -m src.training.train --model scratch_cnn --labels emotion7
python -m src.training.train --model scratch_cnn --labels affect3
python -m src.training.train --model scratch_cnn --train-fraction 0.25 --seed 1
python -m src.training.train --model mobilenetv2 --labels emotion7
python -m src.training.train --model resnet50 --labels affect3

# 4. Evaluate every trained run on the test set, then compare models
python -m src.evaluation.evaluate
python -m src.evaluation.compare
```

Each run saves its best model, config, training history, training curves and a validation
summary to `results/runs/<dataset>_<model>_<labels>_f<fraction>_s<seed>/`. Evaluation adds
`test_metrics.json`, `confusion_matrix.png` and `predictions.png` to the same folder. The comparison
writes `results/tables/comparison/summary.csv` (mean ± std over seeds) and `fine_tuning.csv`, and
figures comparing the models and the effect of fine-tuning to `results/figures/comparison/` (plus a
macro-F1 vs training-data figure when several training fractions were run).

Useful options for the transfer models: `--head-epochs` (default 10), `--fine-tune-lr` (default 1e-5),
`--fine-tune-layers` (0 trains the head only) and `--img-size` (e.g. 128 for faster runs).
If ResNet50 runs out of GPU memory, use `--batch-size 32`.

All commands use FER-2013 by default; add `--dataset rafdb` to use RAF-DB instead.

## Team

| Member | Responsibility |
|---|---|
| Sara Bedulla | Data pipeline, exploratory analysis, CNN from scratch |
| Kejdi Shahu | Transfer learning (MobileNetV2, ResNet50), evaluation and results |

## Ethical note

This project recognises **facial expressions** associated with depression. It does **not**
diagnose depression. Facial expressions alone are not a valid basis for a mental-health
diagnosis, and RAF-DB contains no clinical labels. The models should not be used to make
decisions about individuals.
