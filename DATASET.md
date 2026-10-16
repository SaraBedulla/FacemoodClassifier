# Dataset

## Links

| Dataset | Link | Role in this project |
|---|---|---|
| **FER-2013** | https://www.kaggle.com/datasets/msambare/fer2013 | Main dataset (default in all commands) |
| RAF-DB | https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset (official: http://www.whdeng.cn/RAF/model1.html) | Second dataset with cleaner labels |

Both are public. Downloading from the Kaggle website needs a free Kaggle account. The
[Colab notebook](notebooks/FacemoodClassifier_colab.ipynb) downloads FER-2013 automatically, so no manual
download is needed there.

## FER-2013

FER-2013 (Facial Expression Recognition 2013) was introduced in the ICML 2013 *Challenges in Representation
Learning* facial expression competition (Goodfellow et al., 2013). The images were collected automatically
with Google image search and labelled by one annotator each, so some labels are wrong (human accuracy is
about 65%).

- **35,887** face images, **48×48 pixels, grayscale**, faces roughly centred
- **7 emotion classes**: angry, disgust, fear, happy, sad, surprise, neutral
- Kaggle layout: `train/<emotion>/*.png` and `test/<emotion>/*.png`

### Splits used

The official test set is kept as the test set. A stratified 10% of the official training set is held out for
validation (seed 42), so every model uses exactly the same images.

| Emotion | Train | Validation | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| angry | 3,596 | 399 | 958 | 4,953 | 13.8% |
| disgust | 392 | 44 | 111 | 547 | 1.5% |
| fear | 3,687 | 410 | 1,024 | 5,121 | 14.3% |
| happy | 6,493 | 722 | 1,774 | 8,989 | 25.0% |
| sad | 4,347 | 483 | 1,247 | 6,077 | 16.9% |
| surprise | 2,854 | 317 | 831 | 4,002 | 11.2% |
| neutral | 4,469 | 496 | 1,233 | 6,198 | 17.3% |
| **Total** | **25,838** | **2,871** | **7,178** | **35,887** | |

The classes are imbalanced: *happy* has 16.6× more training images than *disgust*. Training therefore uses
class weights, and results are reported with macro-F1 as well as accuracy.

### Depression-related grouping (`affect3`)

The 7 emotions are also grouped into 3 classes, following findings that depression is associated with more
negative and less positive facial affect:

| Class | Emotions | Train | Validation | Test | Share |
|---|---|---:|---:|---:|---:|
| negative | angry, disgust, fear, sad | 12,022 | 1,336 | 3,340 | 46.5% |
| neutral | neutral | 4,469 | 496 | 1,233 | 17.3% |
| positive | happy, surprise | 9,347 | 1,039 | 2,605 | 36.2% |

## RAF-DB

RAF-DB (Real-world Affective Faces Database; Li et al., 2017) contains 15,339 aligned 100×100 colour face
images with the same 7 emotions. Each image was labelled by about 40 annotators, so its labels are much more
reliable than FER-2013's. Split: 11,043 train / 1,228 validation / 3,068 test. Largest class: happy (38.8%);
smallest: fear (2.3%). The official version is for non-commercial research and requires a request to the
authors; the Kaggle copy above has the same images.

## Preprocessing

`src/data/prepare.py` reads the raw images, keeps each dataset in its native format (FER-2013 48×48×1,
RAF-DB 100×100×3) and saves the fixed splits to `data/processed/<dataset>_splits.npz`. During training,
images are scaled to [0, 1] and augmented with horizontal flip, rotation, zoom and shift (training set only).
Transfer-learning models resize the images to 224×224×3.

## Getting the data locally

```bash
# 1. Download from the Kaggle link above and unzip into data/raw/fer2013/
#    (so you have data/raw/fer2013/train/ and data/raw/fer2013/test/)
# 2. Build the splits
python -m src.data.prepare --dataset fer2013 --source data/raw/fer2013
```

## Ethical note

These datasets contain posed and in-the-wild facial expressions with emotion labels. They contain **no
clinical or depression labels**. This project recognises expressions associated with depression and does not
diagnose depression.

## References

- Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013).
  Challenges in representation learning: A report on three machine learning contests. *International
  Conference on Neural Information Processing (ICONIP)*, 117–124.
- Li, S., Deng, W., & Du, J. (2017). Reliable crowdsourcing and deep locality-preserving learning for expression
  recognition in the wild. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2852–2861.
