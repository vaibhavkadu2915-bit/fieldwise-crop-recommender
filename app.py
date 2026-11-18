"""Streamlit interface for soil-based crop recommendations."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_model.joblib"

st.set_page_config(page_title="Fieldwise | Crop guide", page_icon="🌱", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --ink:#19372c; --muted:#718078; --green:#28734f; --lime:#d8edb7; --paper:#f7f8f3; }
      .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
      .block-container { max-width:1120px; padding:2rem 2.2rem 3rem; }
      header[data-testid="stHeader"] { background:transparent; }
      #MainMenu, footer { visibility:hidden; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
      .brand { color:var(--green); font-weight:700; letter-spacing:.14em; font-size:.76rem; text-transform:uppercase; margin-bottom:.65rem; }
      .hero { background:linear-gradient(120deg,#e8f1df 0%,#f2f4e8 65%,#e7eee3 100%); border:1px solid #dce6d7; border-radius:22px; padding:2rem 2.25rem; margin:.2rem 0 1.5rem; }
      .hero h1 { font-size:2.35rem; line-height:1.15; margin:0 0 .6rem; }
      .hero p { color:#52685d; font-size:1.02rem; max-width:650px; margin:0; }
      .section-label { text-transform:uppercase; letter-spacing:.11em; color:#6b7b70; font-size:.72rem; font-weight:700; margin:.35rem 0 .8rem; }
      div[data-testid="stForm"] { background:#fff; border:1px solid #e5e9e1; border-radius:20px; padding:1.25rem 1.4rem .5rem; box-shadow:0 8px 24px rgba(34,66,49,.045); }
      div[data-testid="stNumberInput"] label { font-weight:600; color:#304a3d; }
      div[data-testid="stNumberInput"] input { border-radius:11px; }
      div[data-testid="stFormSubmitButton"] button { background:#28734f; border:0; color:#fff; border-radius:11px; font-weight:700; padding:.68rem 1.1rem; }
      div[data-testid="stFormSubmitButton"] button:hover { background:#1e5e3e; color:#fff; }
      .result-card { background:#193d30; color:#f8fff9; border-radius:20px; padding:1.45rem 1.6rem; margin-top:1rem; }
      .result-kicker { color:#c3d7c9; font-size:.76rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
      .result-crop { color:#fff; font:800 2.1rem 'Manrope',sans-serif; text-transform:capitalize; margin:.35rem 0 .2rem; }
      .result-note { color:#d6e4da; font-size:.9rem; }
      .tip { color:#718078; font-size:.82rem; line-height:1.5; margin-top:1rem; }
      .stProgress > div > div > div > div { background:#73a877; }
      div[data-testid="stDataFrame"] { border:1px solid #e5e9e1; border-radius:12px; overflow:hidden; }
      @media(max-width:700px) { .block-container{padding:1.1rem 1rem 2rem}.hero{padding:1.5rem}.hero h1{font-size:1.8rem} }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="brand">FIELDWISE&nbsp; / &nbsp;SOIL-BASED CROP GUIDE</div>
      <h1>A better starting point<br>for your next planting.</h1>
      <p>Share the results of your soil test. The model will compare them with crop patterns in the training data and suggest a likely match.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.warning("The recommendation model is not trained yet. Run `python src/train_model.py` to create it.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

left, right = st.columns([1.05, .95], gap="large")
with left:
    st.markdown('<div class="section-label">01 &nbsp; Your soil test</div>', unsafe_allow_html=True)
    st.write("Enter the values exactly as shown on your lab report.")
    with st.form("soil_values"):
        first, second = st.columns(2, gap="medium")
        with first:
            n = st.number_input("Nitrogen · N", min_value=0.0, value=50.0, step=1.0, help="Use the same units as the training dataset.")
            p = st.number_input("Phosphorous · P", min_value=0.0, value=50.0, step=1.0, help="Use the same units as the training dataset.")
        with second:
            k = st.number_input("Potassium · K", min_value=0.0, value=50.0, step=1.0, help="Use the same units as the training dataset.")
            ph = st.number_input("Soil acidity · pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        submitted = st.form_submit_button("Find a crop match  →", use_container_width=True)
    st.markdown('<div class="tip">Tip: soil values and units should match the dataset used to train this model. Local climate, water availability, season, and soil texture also affect what grows well.</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-label">02 &nbsp; Recommendation</div>', unsafe_allow_html=True)
    if submitted:
        values = pd.DataFrame([[n, p, k, ph]], columns=features)
        crop = model.predict(values)[0]
        st.markdown(
            f'<div class="result-card"><div class="result-kicker">Best model match</div><div class="result-crop">{crop}</div><div class="result-note">Based on the soil measurements you entered</div></div>',
            unsafe_allow_html=True,
        )
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(values)[0]
            ranking = sorted(zip(model.classes_, probabilities), key=lambda row: row[1], reverse=True)[:3]
            st.markdown("#### Other possible matches")
            for name, probability in ranking[1:]:
                st.markdown(f"**{name.title()}** <span style='color:#718078'> · {probability:.0%}</span>", unsafe_allow_html=True)
                st.progress(float(probability))
            st.caption("Percentages show model scores, not guaranteed chances of success.")
    else:
        st.markdown(
            '<div class="result-card"><div class="result-kicker">Ready when you are</div><div class="result-crop">Your field, your data.</div><div class="result-note">Complete the soil test on the left to see the model’s crop matches here.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("#### What this tool considers")
        st.write("Nitrogen, phosphorous, potassium, and soil pH are compared together by a multinomial logistic regression model.")

st.divider()
st.caption("Fieldwise is an educational decision-support demo based on a benchmark dataset. Use local agricultural guidance before making planting decisions.")
