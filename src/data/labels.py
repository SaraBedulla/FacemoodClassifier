"""Class names and the two label schemes used in the project."""
import numpy as np

# Canonical 7-class order (same as the label ids in the original fer2013.csv).
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Grouping used for the depression-related task: depression is associated with
# more negative affect and less positive affect, so we collapse the 7 emotions
# into negative / neutral / positive expression patterns.
AFFECT_CLASSES = ["negative", "neutral", "positive"]
EMOTION_TO_AFFECT = {
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    "neutral": "neutral",
    "happy": "positive",
    "surprise": "positive",
}

LABEL_SCHEMES = {
    "emotion7": EMOTIONS,
    "affect3": AFFECT_CLASSES,
}


def to_scheme(y, scheme):
    """Convert 7-class emotion labels to the requested label scheme."""
    if scheme == "emotion7":
        return y
    if scheme == "affect3":
        mapping = np.array([AFFECT_CLASSES.index(EMOTION_TO_AFFECT[e]) for e in EMOTIONS])
        return mapping[y]
    raise ValueError(f"Unknown label scheme {scheme!r}; choose from {list(LABEL_SCHEMES)}")
