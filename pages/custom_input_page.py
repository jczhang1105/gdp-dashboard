import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from collections import Counter

# ------------------------
# 页面美化：黑金色背景 + 玻璃拟态风格
# ------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000, #2c2c2c);
    color: #FFD700;
    font-family: 'Segoe UI', sans-serif;
}

.css-18e3th9, .css-1d391kg {
    background-color: rgba(255, 215, 0, 0.05);
    padding: 2rem;
    border-radius: 1rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 32px 0 rgba(255, 215, 0, 0.15);
    border: 1px solid rgba(255, 215, 0, 0.1);
}

.stTextInput > div > input,
.stTextArea > div > textarea {
    background-color: rgba(255,255,255,0.08);
    color: #FFD700;
    border-radius: 8px;
    border: 1px solid rgba(255,215,0,0.3);
}

.stButton > button {
    background: linear-gradient(135deg, #FFD700, #DAA520);
    color: #000;
    font-weight: 600;
    padding: 0.5rem 1rem;
    border-radius: 10px;
    transition: all 0.3s ease;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #DAA520, #FFD700);
    color: #111;
}

h1, h2, h3, h4, h5, h6, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# 配置路径与参数
# ------------------------
MODEL_PATH = Path("final_optimized_model.joblib")
DATA_PATH = Path("collected_data.xlsx")
SHEET_NAMES = ["新细菌", "python"]

# 原始特征顺序（与模型训练时保持一致）
FEATURE_COLS = [
    'bp', 'AAT', 'ACC', 'ACT', 'AGA', 'AGG', 'CAC', 'CCA', 'CCG', 'CGG',
    'CTG', 'GGG', 'TAC', 'TGA', 'TTC', 'Wavelength(nm)', 'Dose(mJ/cm2)',
    'Temp', 'Ph', 'Log N0', 'AG', 'AT', 'GT', 'TC', 'AT content',
    "Gram-positive_bacteria(Y/N)", "Spore(Y/N)"
]

# ------------------------
# K-mer 特征提取函数
# ------------------------
def extract_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmer_feature_vector(sequence, k, all_kmers):
    freq = Counter(extract_kmers(sequence, k))
    return [freq.get(kmer, 0) for kmer in all_kmers]

# ------------------------
# 加载模型
# ------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"❌ 模型文件未找到: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

# ------------------------
# 自定义输入特征构建
# ------------------------
def compute_features_from_sequence(seq):
    k = 3
    all_possible_kmers = sorted([a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"])
    kmer_features = kmer_feature_vector(seq.upper(), k, all_possible_kmers)
    kmer_dict = dict(zip(all_possible_kmers, kmer_features))

    features = {
        **kmer_dict,
        'bp': len(seq),
        'AT content': (seq.upper().count('A') + seq.upper().count('T')) / len(seq),
        'Gram-positive_bacteria(Y/N)': 1,
        'Spore(Y/N)': 0,
        'AG': seq.upper().count('AG'),
        'AT': seq.upper().count('AT'),
        'GT': seq.upper().count('GT'),
        'TC': seq.upper().count('TC'),
    }

    return pd.Series(features)

# ------------------------
# 自定义输入页面
# ------------------------
st.set_page_config(page_title="Custom UV Prediction", layout="centered")
st.title("✨ UV Inactivation Prediction from 16S rRNA")
st.markdown("Upload or paste your 16S rRNA sequence and set environmental parameters to estimate UV disinfection efficiency.")

sequence = st.text_area("🧬 Enter 16S rRNA Sequence", height=180)
wavelength = st.slider("🔆 Set UV Wavelength (nm)", 200, 300, 254)
ph = st.slider("🧪 Set pH Value", 5.5, 9.0, 7.0)
temp = st.slider("🌡 Set Temperature (°C)", 5, 50, 25)
dose = st.slider("⚡ Set UV Dose (mJ/cm²)", 0.1, 120.0, 10.0)

if st.button("🚀 Predict Inactivation"):
    if len(sequence.strip()) < 30:
        st.warning("⚠️ Please enter a valid 16S rRNA sequence with at least 30 bases.")
        st.stop()

    model = load_model()
    base_features = compute_features_from_sequence(sequence)

    base_features["Wavelength(nm)"] = wavelength
    base_features["Dose(mJ/cm2)"] = dose
    base_features["Temp"] = temp
    base_features["Ph"] = ph
    base_features["Log N0"] = 6.0

    for col in FEATURE_COLS:
        if col not in base_features:
            base_features[col] = 0

    input_df = pd.DataFrame([base_features])[FEATURE_COLS]

    st.subheader("📊 Computed Input Feature Summary")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    try:
        y_pred = model.predict(input_df)[0]
        st.success(f"✅ Predicted Inactivation (log N₀/Nₜ): **{y_pred:.2f}**")
        st.markdown("A higher log reduction means better UV inactivation performance.")
    except Exception as e:
        st.error("❌ Prediction failed due to internal error.")
        st.exception(e)
else:
    st.info("⬅️ Enter your 16S rRNA sequence and click Predict to continue.")
