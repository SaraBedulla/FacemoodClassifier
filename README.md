# FacemoodClassifier

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
| MobileNetV2 | 224×224×3 | ImageNet-pretrained, frozen backbone, then fine-tuned |
| ResNet50 | 224×224×3 | ImageNet-pretrained, frozen backbone, then fine-tuned |

All models use the same data splits, augmentation (flip, rotation, zoom, shift), class weighting,
and evaluation, so the comparison is fair.

## Experiments

1. **Main comparison**: all three models on `emotion7` and `affect3`.
2. **Training-data size**: each model trained on 10%, 25%, 50% and 100% of the training set, to see
   where transfer learning helps most.
3. Every configuration is run with several random seeds and reported as mean ± std.

**Metrics:** accuracy, macro-F1 (the classes are imbalanced), per-class recall, confusion matrix,
number of parameters, training time, and inference time per image.

## Project structure

```
FacemoodClassifier/
├── src/
│   ├── config/
│   │   └── config.py            # paths and constants
│   ├── data/
│   │   ├── labels.py            # emotion7 / affect3 label schemes
│   │   ├── prepare.py           # reads raw RAF-DB / FER-2013, builds fixed splits
│   │   ├── dataset.py           # loading splits, class weights, tf.data + augmentation
│   │   └── eda.py               # exploratory data analysis
│   ├── models/
│   │   ├── registry.py          # list of models available for training
│   │   ├── scratch_cnn.py       # CNN from scratch
│   │   └── transfer.py          # MobileNetV2 / ResNet50
│   ├── training/
│   │   └── train.py             # trains one model configuration
│   ├── evaluation/              # test-set metrics and comparison of runs
│   └── visualization/
│       └── plots.py             # shared figure style
├── requirements.txt
└── README.md
```

Created when you run the code (not committed):

```
data/
├── raw/                         # datasets as downloaded (e.g. raw/rafdb/)
└── processed/                   # fixed train/val/test splits
results/
├── figures/                     # EDA figures
├── tables/                      # EDA class-count tables
└── runs/                        # one folder per training run
```

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
```

Each run saves its best model, config, training history, training curves and a validation
summary to `results/runs/<dataset>_<model>_<labels>_f<fraction>_s<seed>/`.

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
