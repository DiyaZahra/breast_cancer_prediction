import json
from pathlib import Path

import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

ART = Path(__file__).parent / "artifacts"

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(ART / "model.json")
    meta = json.loads((ART / "meta.json").read_text())
    examples = {
        "Typical benign": json.loads((ART / "example_benign.json").read_text()),
        "Typical malignant": json.loads((ART / "example_malignant.json").read_text()),
    }
    return model, meta, examples


model, meta, examples = load_model()
features, stats, metrics = meta["features"], meta["stats"], meta["metrics"]
GROUPS = {"Mean": "_mean", "Standard error": "_se", "Worst": "_worst"}


def predict(df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df[features])[:, 1]
    out = df.copy()
    out["malignant_probability"] = proba
    out["prediction"] = ["Malignant" if p >= 0.5 else "Benign" for p in proba]
    return out


def load_example(name: str):
    for f in features:
        st.session_state[f] = float(examples[name][f])


# ---------- Header ----------
st.title("🩺 Breast Cancer Tumor Classifier")
st.caption("XGBoost model trained on the Wisconsin Diagnostic Breast Cancer dataset "
           "(569 samples, 30 cell-nucleus measurements from fine-needle aspirate images).")
st.warning("**Educational demo only — not a medical device.** Do not use it to make health "
           "decisions. Anyone with health concerns should see a qualified clinician.")

with st.sidebar:
    st.header("Model performance")
    st.metric("Hold-out ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Hold-out accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("5-fold CV ROC-AUC", f"{metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
    st.caption(f"Hold-out set: {metrics['n_test']} samples never used for training or tuning.")
    st.divider()
    st.subheader("Top features")
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(8)
    st.bar_chart(imp)

tab_single, tab_batch = st.tabs(["Single prediction", "Batch (CSV upload)"])

# ---------- Single prediction ----------
with tab_single:
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("Load benign example", on_click=load_example, args=("Typical benign",))
    c2.button("Load malignant example", on_click=load_example, args=("Typical malignant",))

    for f in features:  # initialise defaults once
        st.session_state.setdefault(f, float(examples["Typical benign"][f]))

    values = {}
    sub_tabs = st.tabs(list(GROUPS))
    for tab, (label, suffix) in zip(sub_tabs, GROUPS.items()):
        with tab:
            cols = st.columns(3)
            group_feats = [f for f in features if f.endswith(suffix)]
            for i, f in enumerate(group_feats):
                lo, hi = stats[f]["min"], stats[f]["max"]
                values[f] = cols[i % 3].number_input(
                    f.removesuffix(suffix).replace("_", " ").title(),
                    min_value=0.0, max_value=float(hi * 2), key=f,
                    step=float((hi - lo) / 200) or 0.001, format="%.5f",
                    help=f"Training range: {lo:.4g} – {hi:.4g}",
                )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([values])
        res = predict(row).iloc[0]
        p = float(res["malignant_probability"])
        left, right = st.columns([1, 2])
        with left:
            if res["prediction"] == "Malignant":
                st.error(f"### Predicted: Malignant\n{p:.1%} probability")
            else:
                st.success(f"### Predicted: Benign\n{1 - p:.1%} probability")
        with right:
            st.write("Malignant probability")
            st.progress(p)
            out_of_range = [f for f in features
                            if not stats[f]["min"] <= values[f] <= stats[f]["max"]]
            if out_of_range:
                st.info(f"{len(out_of_range)} input(s) fall outside the training range, "
                        "so this prediction may be less reliable.")

# ---------- Batch ----------
with tab_batch:
    st.write("Upload a CSV with the 30 feature columns (same names as the Kaggle dataset). "
             "Extra columns such as `id` or `diagnosis` are ignored.")
    template = pd.DataFrame([examples["Typical benign"]])[features]
    st.download_button("Download CSV template", template.to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            result = predict(data)
            st.dataframe(result[["prediction", "malignant_probability"]
                                + [c for c in data.columns if c not in features]
                                + features[:5]], use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")
