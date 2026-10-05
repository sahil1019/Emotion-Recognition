import numpy as np
import pandas as pd
import streamlit as st

from predict import predict_features, load_samples, MODALITIES

st.set_page_config(page_title="Multimodal Emotion Recognition")
st.title("Multimodal Emotion Recognition")
st.caption(
    "Fuses audio, facial and physiological features with a RandomForest. "
    "This demo works on pre-extracted feature vectors, not raw audio or video."
)


@st.cache_resource
def get_samples():
    return load_samples()


DATA, LABELS, N = get_samples()

tab1, tab2 = st.tabs(["Try a dataset sample", "Upload feature files"])

with tab1:
    idx = st.slider("Sample index", 0, max(N - 1, 0), 0)
    if st.button("Predict", key="sample"):
        feats = {m: DATA[m][idx:idx + 1] for m in MODALITIES}
        labels, proba, classes = predict_features(feats)
        true = str(LABELS[idx]) if LABELS is not None else "unknown"
        st.success(f"Predicted: {labels[0]}  |  Dataset label: {true}")
        st.bar_chart(pd.Series(proba[0], index=classes))

with tab2:
    st.write("Upload `.npy` files of raw features (rows = samples), same format as the training data.")
    a = st.file_uploader("Audio features (.npy)", type="npy")
    f = st.file_uploader("Facial features (.npy)", type="npy")
    p = st.file_uploader("Physiological features (.npy)", type="npy")
    if st.button("Predict", key="upload"):
        try:
            feats = {
                "audio": np.load(a, allow_pickle=True) if a else None,
                "face": np.load(f, allow_pickle=True) if f else None,
                "phys": np.load(p, allow_pickle=True) if p else None,
            }
            labels, proba, classes = predict_features(feats)
            df = pd.DataFrame(proba, columns=classes).round(3)
            df.insert(0, "prediction", labels)
            st.dataframe(df)
        except Exception as e:
            st.error(str(e))
