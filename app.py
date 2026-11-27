"""Streamlit web app.  Run:  streamlit run app.py   (after python train.py)"""
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cancer Risk Predictor", page_icon="🩺")
model = joblib.load("best_model.joblib")
meta = json.load(open("meta.json"))

st.title("🩺 Breast Cancer Risk Predictor")
st.caption(f"Best model selected automatically: **{meta['best_model']}**. "
           "Educational demo only - not a medical device.")

st.sidebar.header("Tumor measurements")
values = {f: meta["stats"][f]["mean"] for f in meta["features"]}   # defaults
for f in meta["top_features"]:
    s = meta["stats"][f]
    values[f] = st.sidebar.slider(f, s["min"], s["max"], s["mean"])

if st.button("Predict", type="primary"):
    row = pd.DataFrame([values])[meta["features"]]
    p_benign = float(model.predict_proba(row)[0, 1])
    if p_benign >= 0.5:
        st.success(f"Likely BENIGN  (confidence {p_benign:.1%})")
    else:
        st.error(f"Likely MALIGNANT  (confidence {1 - p_benign:.1%})")
    st.progress(p_benign, text=f"P(benign) = {p_benign:.1%}")

with st.expander("Model comparison"):
    st.dataframe(pd.read_csv("model_results.csv"), hide_index=True)
    st.image("results.png")
