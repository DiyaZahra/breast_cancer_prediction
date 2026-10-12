# Breast Cancer Tumor Classifier

A machine learning project that predicts whether a breast tumor is **benign** or **malignant** from
measurements of cell nuclei, with a Streamlit web app for trying the model interactively. 
live link : https://breastcancerprediction-nb856xynfasmgqfrwvtthb.streamlit.app/

> **Educational project only — not a medical device.** Predictions must not be used for diagnosis
> or health decisions.
>

## Dataset

[Wisconsin Diagnostic Breast Cancer (WDBC)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic):
569 samples, 30 numeric features computed from digitized images of fine-needle aspirates of breast masses.

- **Target:** `diagnosis` — 1 = Malignant (212 samples), 0 = Benign (357 samples)
- **Features:** 10 nucleus characteristics (radius, texture, perimeter, area, smoothness, compactness,
  concavity, concave points, symmetry, fractal dimension), each reported as the **mean**, **standard
  error (se)** and **worst** (mean of the three largest values) value, giving 30 features.
- No missing values; the `id` and empty `Unnamed: 32` columns in the Kaggle CSV are dropped.

## Approach

1. **Cleaning:** drop identifier/empty columns and encode the label (M → 1, B → 0).
2. **Splitting:** stratified 80/20 train/test split. The training portion is split again to get a
   validation set used only for early stopping, so the test set is never used for tuning.
3. **Model:** `XGBClassifier` (gradient-boosted trees) with `learning_rate=0.05`, `max_depth=4`,
   `subsample=0.8`, `colsample_bytree=0.8`, and early stopping on validation AUC (167 trees selected).
   Tree models need no feature scaling.
4. **Evaluation:** hold-out test set plus 5-fold stratified cross-validation.
5. **Final model:** refit on all 569 samples with the selected number of trees and saved for the app.

## Results

| Metric | Value |
|---|---|
| Hold-out accuracy (114 samples) | 98.2% |
| Hold-out ROC-AUC | 0.995 |
| 5-fold CV ROC-AUC | 0.993 ± 0.005 |

Hold-out confusion matrix (rows = actual, columns = predicted):

| | Predicted benign | Predicted malignant |
|---|---|---|
| **Actual benign** | 72 | 0 |
| **Actual malignant** | 2 | 40 |

The dataset is small and clean, so results on real-world data would likely be lower.

## The app

- Enter the 30 measurements by hand (grouped into Mean / Standard error / Worst tabs), or load a typical
  benign or malignant example.
- See the predicted class and malignant probability, with a notice when inputs fall outside the range
  seen in training.
- Upload a CSV for batch predictions and download the results.

## Usage

```bash
pip install -r requirements.txt
python train.py          # retrain and regenerate artifacts/ (optional, artifacts are included)
streamlit run app.py
```

Train on your own copy of the Kaggle CSV with `python train.py --csv "breast cancer.csv"`.
