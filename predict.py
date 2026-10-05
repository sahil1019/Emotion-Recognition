import os
import numpy as np
import joblib
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ART = joblib.load(os.path.join(HERE, "multimodal_emotion_artifact.joblib"))
MODEL = ART["model_rf"]
SCALERS = ART["per_mod_scalers"]
SELECTORS = ART["per_mod_selectors"]
PCAS = ART.get("per_mod_pcas", {}) or {}
MODALITIES = [name for (name, _, _) in ART["col_splits"]]  # order matters

# The saved artifact did not include the PCA used for the face modality, so it was
# rebuilt from the training data and stored as plain arrays (pca_face.npz).
NPZ_PCAS = {}
for _name in MODALITIES:
    _p = os.path.join(HERE, f"pca_{_name}.npz")
    if os.path.exists(_p):
        _z = np.load(_p)
        NPZ_PCAS[_name] = {"mean": _z["mean"], "components": _z["components"]}


# ---- same augmentation functions as in training ----
def _as_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def augment_audio(X):
    X = _as_2d(X)
    return np.hstack([
        X,
        X.mean(axis=1, keepdims=True),
        X.std(axis=1, keepdims=True),
        X.min(axis=1, keepdims=True),
        X.max(axis=1, keepdims=True),
        st.skew(X, axis=1).reshape(-1, 1),
        st.kurtosis(X, axis=1).reshape(-1, 1),
    ])


def augment_face(X):
    X = _as_2d(X)
    return np.hstack([
        X,
        X.mean(axis=1, keepdims=True),
        X.std(axis=1, keepdims=True),
        np.percentile(X, 25, axis=1).reshape(-1, 1),
        np.percentile(X, 75, axis=1).reshape(-1, 1),
    ])


def augment_phys(X):
    X = _as_2d(X)
    return np.hstack([
        X,
        X.mean(axis=1, keepdims=True),
        X.std(axis=1, keepdims=True),
        X.min(axis=1, keepdims=True),
        X.max(axis=1, keepdims=True),
        np.median(X, axis=1, keepdims=True),
    ])


AUG = {"audio": augment_audio, "face": augment_face, "phys": augment_phys}


def _prepare(name, X):
    """Augment raw features, unless they already have the width the scaler expects."""
    X = _as_2d(X)
    expected = SCALERS[name].n_features_in_
    if X.shape[1] == expected:
        return X  # already augmented
    X_aug = AUG[name](X)
    if X_aug.shape[1] != expected:
        raise ValueError(
            f"'{name}' features have {X.shape[1]} columns; the model expects "
            f"{expected} after augmentation (raw input should have "
            f"{expected - (X_aug.shape[1] - X.shape[1])} columns)."
        )
    return X_aug


def predict_features(feats):
    """feats: dict {'audio': (n,d), 'face': (n,d), 'phys': (n,d)}.
    Returns (labels, probabilities, class_names)."""
    parts = []
    for name in MODALITIES:
        if name not in feats or feats[name] is None:
            raise ValueError(f"Missing modality '{name}'. This model needs: {MODALITIES}")
        X = _prepare(name, feats[name])
        X = SCALERS[name].transform(X)
        if name in PCAS:
            X = PCAS[name].transform(X)
        elif name in NPZ_PCAS:
            X = (X - NPZ_PCAS[name]["mean"]) @ NPZ_PCAS[name]["components"].T
        X = SELECTORS[name].transform(X)
        parts.append(X)
    n = min(p.shape[0] for p in parts)
    X = np.hstack([p[:n] for p in parts])
    proba = MODEL.predict_proba(X)
    labels = MODEL.classes_[proba.argmax(axis=1)]
    return labels, proba, list(MODEL.classes_)


# ---- sample data for the demo tab ----
def load_samples():
    def ld(f):
        p = os.path.join(HERE, f)
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None

    data = {
        "audio": ld("F_audio_full.npy"),
        "face": ld("F_facial_full.npy"),
        "phys": ld("F_physio.npy"),
    }
    labels = ld("Y_audio_full.npy")
    if labels is None:
        labels = ld("Y_facial_full.npy")
    avail = [data[m] for m in MODALITIES if data.get(m) is not None]
    n = min(a.shape[0] for a in avail) if avail else 0
    if labels is not None:
        n = min(n, len(labels))
    return data, labels, n
