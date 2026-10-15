# FacemoodClassifier

Depression-related facial expression classification with deep learning.

This project classifies facial-expression images into emotions and into depression-related
expression patterns. It compares a **CNN trained from scratch** with **pretrained transfer-learning
models (MobileNetV2 and ResNet50)**.

## Research question

> How does a CNN trained from scratch compare with transfer learning (MobileNetV2, ResNet50)
> for facial expression classification, in terms of accuracy, efficiency, and the amount of
> training data needed?

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013): about 35,000 grayscale face images
(48×48 px) labelled with 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.

The dataset is not included in this repository. Download it from Kaggle and place it in `data/raw/`.

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
| Scratch CNN | 48×48×1 | 4 convolutional blocks (Conv–BN–ReLU ×2, max-pool, dropout) + dense head |
| MobileNetV2 | 96×96×3 | ImageNet-pretrained, frozen backbone, then fine-tuned |
| ResNet50 | 96×96×3 | ImageNet-pretrained, frozen backbone, then fine-tuned |

All models use the same data splits, augmentation (flip, rotation, zoom, shift), class weighting,
and evaluation, so the comparison is fair.

## Experiments

1. **Main comparison**: all three models on `emotion7` and `affect3`.
2. **Training-data size**: each model trained on 10%, 25%, 50% and 100% of the training set, to see
   where transfer learning helps most.
3. Every configuration is run with several random seeds and reported as mean ± std.

**Metrics:** accuracy, macro-F1 (FER-2013 is imbalanced), per-class recall, confusion matrix,
number of parameters, training time, and inference time per image.

## Project structure

```
FacemoodClassifier/
├── data/
│   ├── raw/                 # FER-2013 as downloaded (not committed)
│   └── processed/           # fixed train/val/test splits (not committed)
├── src/
│   ├── data.py              # loading, splits, label schemes, augmentation, class weights
│   ├── eda.py               # exploratory data analysis figures
│   ├── models/
│   │   ├── scratch_cnn.py   # CNN from scratch
│   │   └── transfer.py      # MobileNetV2 / ResNet50
│   ├── train.py             # trains one model configuration
│   ├── evaluate.py          # test-set metrics and summary across runs
│   └── plots.py             # shared figure style
├── results/
│   ├── figures/
│   └── runs/                # one folder per training run
├── requirements.txt
└── README.md
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
python -m src.data --source data/raw/fer2013        # folder version
python -m src.data --source data/raw/fer2013.csv    # or the CSV version

# 2. Explore the data
python -m src.eda

# 3. Train and evaluate a model
python -m src.train --model scratch_cnn --labels emotion7
python -m src.train --model scratch_cnn --labels affect3
python -m src.train --model scratch_cnn --train-fraction 0.25 --seed 1

# 4. Collect the results of all runs into one table
python -m src.evaluate --summary
```

Each run saves its model, training curves, confusion matrix, classification report and metrics
to `results/runs/<model>_<labels>_f<fraction>_s<seed>/`.

## Team

| Member | Responsibility |
|---|---|
| Sara Bedulla | Data pipeline, exploratory analysis, CNN from scratch |
| Kejdi Shahu | Transfer learning (MobileNetV2, ResNet50), evaluation and results |

## Ethical note

This project recognises **facial expressions** associated with depression. It does **not**
diagnose depression. Facial expressions alone are not a valid basis for a mental-health
diagnosis, and FER-2013 contains no clinical labels. The models should not be used to make
decisions about individuals.
