# 🛵 Zomato Delivery Time Predictor

A Streamlit web app that predicts food delivery time using a Random Forest
model trained on the Zomato delivery operations dataset.

## Files

- `app.py` — the Streamlit app
- `requirements.txt` — Python dependencies
- `Delivery_Time_Prediction.pkl` — your trained model (you already have this)

## ⚠️ Important note about accuracy

Your original notebook only saved the trained `RandomForestRegressor` with
`joblib.dump(rf, ...)` — it did **not** save the `ColumnTransformer` (`ct`)
that encoded your categorical columns. `app.py` rebuilds that encoding by
hand, matching your code exactly (same category lists, same
`OrdinalEncoder`/`OneHotEncoder(drop='first')` logic, same column order).

This works well because your dataset is the well-known Zomato Delivery
Operations dataset with fixed category values. But for a bulletproof,
production-grade version, add this to the **end of your Colab notebook** and
re-download the new file:

```python
import joblib
joblib.dump(ct, 'preprocessor.pkl')
```

Then upload `preprocessor.pkl` to GitHub alongside the model, and in
`app.py` replace the manual `build_feature_vector` step with
`ct.transform(input_dataframe)`. I’m happy to update the app for you if you
generate this file.

## 🚀 Deploy entirely from your iPad (no laptop needed)

### 1. Create a GitHub repo

1. Open **github.com** in Safari, sign in (or create a free account).
1. Tap **+ → New repository**. Name it e.g. `zomato-delivery-time-predictor`.
1. Make it **Public** (required for the free Streamlit Cloud tier).

### 2. Upload the files

1. In your new repo, tap **Add file → Upload files**.
1. Upload `app.py`, `requirements.txt`, and `Delivery_Time_Prediction.pkl`.
1. Tap **Commit changes**.

### 3. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub.
1. Tap **Create app → From existing repo**.
1. Select your repo, branch `main`, and main file path `app.py`.
1. Tap **Deploy**. Build takes ~2–3 minutes.
1. You’ll get a public link like:
   `https://your-app-name.streamlit.app`

That link is what you post to LinkedIn.

## 📸 Getting good screenshots for LinkedIn

- Open your deployed app link on your iPad in Safari.
- Fill in a realistic example (e.g. rush-hour traffic + rain) so the
  prediction card is visible with a real number.
- Use the **Share → Screenshot** / full-page capture, or Safari’s
  “Full Page” screenshot option (tap the screenshot thumbnail after
  taking it, then “Full Page” tab) for a clean, complete capture.
- Post 2–3 images: the input form, the prediction result card, and the map view.
- Suggested LinkedIn caption angle: mention the problem (delivery time
  estimation), your approach (EDA → feature engineering incl. Haversine
  distance → Random Forest, R² achieved), and the live app link.

## Running locally (optional, if you ever get a laptop)

```bash
pip install -r requirements.txt
streamlit run app.py
```