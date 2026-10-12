# Breast Cancer Tumor Classifier

Streamlit app around an XGBoost model trained on the Wisconsin Diagnostic Breast Cancer dataset.
**Educational demo only — not a medical device.**

## Run locally
```bash
pip install -r requirements.txt
python train.py                 # optional: retrain (artifacts/ is already included)
streamlit run app.py
```
To train on your own CSV: `python train.py --csv "breast cancer.csv"`.

## Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and push this folder (include `artifacts/`).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.

## Deploy with Docker (Hugging Face Spaces, Render, etc.)
```bash
docker build -t bc-app . && docker run -p 8501:8501 bc-app
```
