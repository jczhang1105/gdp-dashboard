import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

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

# 支持的菌种选项
STRAIN_MAPPING = {
    "CP118080": "CP118080",
    "CP012951.1": "CP012951.1",
    "ATCC 27562": "ATCC 27562",
    "ATCC 51132": "ATCC 51132",
    "ATCC 11229": "ATCC 11229",
    "IFO 3301": "IFO 3301",
    "ATCC 10145": "ATCC 10145",
    "ATCC 27853": "ATCC 27853",
    "ATCC 35855": "ATCC 35855",
    "CGMCC 1.2456": "CGMCC 1.2456",
    "ATCC 25923": "ATCC 25923",
    "ATCC 27142": "ATCC 27142",
    "ATCC 6633": "ATCC 6633",
    "ATCC 33152": "ATCC 33152",
    "ATCC 33462": "ATCC 33462",
    "ATCC 15442": "ATCC 15442",
    "ATCC 25922": "ATCC 25922",
    "ATCC 29425": "ATCC 29425",
    "ATCC 33823": "ATCC 33823",
    "CGMCC 1.3373": "CGMCC 1.3373",
    "ATCC 15755": "ATCC 15755",
    "ATCC 6633 (Spore=1)": "ATCC 6633",
    "ATCC 6633 (Spore=0)": "ATCC 6633"
}

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
# 加载模板数据（多个 sheet）
# ------------------------
@st.cache_data
def load_template_data():
    if not DATA_PATH.exists():
        st.error(f"❌ 数据文件未找到: {DATA_PATH}")
        st.stop()
    combined = []
    for sheet in SHEET_NAMES:
        df = pd.read_excel(DATA_PATH, sheet_name=sheet)
        combined.append(df)
    return pd.concat(combined, ignore_index=True)

# ------------------------
# 获取菌种特征
# ------------------------
def get_strain_features(strain_name):
    df = load_template_data()

    if "ATCC 6633" in strain_name:
        spore_flag = 1 if "(Spore=1)" in strain_name else 0
        filtered = df[(df["细菌/微生物"] == "ATCC 6633") & (df["Spore(Y/N)"] == spore_flag)]
    else:
        filtered = df[df["细菌/微生物"] == strain_name]

    if filtered.empty:
        st.error(f"⚠️ 未在数据中找到菌种：{strain_name}")
        st.stop()

    try:
        first_row = filtered.iloc[0]
        values = [first_row.get(col, np.nan) for col in FEATURE_COLS]
        return pd.Series(data=values, index=FEATURE_COLS)
    except Exception as e:
        st.error(f"⚠️ 构建菌种特征数据失败：{e}")
        st.stop()

# ------------------------
# 页面设置
# ------------------------
st.set_page_config(page_title="UV Inactivation Predictor", page_icon="🧫", layout="centered")
st.title("💡 UV Inactivation Prediction Tool")

# ------------------------
# 侧边栏输入
# ------------------------
st.sidebar.header("🔬 Inactivation Parameters")
strain = st.sidebar.selectbox("Select Strain", list(STRAIN_MAPPING.keys()))
wavelength = st.sidebar.slider("Wavelength (nm)", 200, 300, 254)
ph = st.sidebar.slider("pH", 5.5, 9.0, 7.2)
temp = st.sidebar.slider("Temperature (°C)", 5, 50, 25)
dose = st.sidebar.slider("Dose (mJ/cm²)", 0.1, 120.0, 10.0)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔍 Predict Inactivation")

# ------------------------
# 绘图函数（添加容错）
# ------------------------
def plot_dose_response(model, base_features):
    doses = np.linspace(0.1, 120, 100)
    predictions = []

    for d in doses:
        features = base_features.copy()
        features["Dose(mJ/cm2)"] = d
        input_df = pd.DataFrame([features])

        # 对齐模型特征
        try:
            model_features = model.feature_names_
        except AttributeError:
            model_features = FEATURE_COLS

        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df.reindex(columns=model_features)

        try:
            y = model.predict(input_df)[0]
            predictions.append(y)
        except:
            predictions.append(np.nan)

    # 平滑曲线
    doses_smooth = np.linspace(doses.min(), doses.max(), 300)
    spline = make_interp_spline(doses, predictions, k=3)
    predictions_smooth = spline(doses_smooth)

    fig, ax = plt.subplots()
    ax.plot(doses_smooth, predictions_smooth, color='darkblue')
    ax.set_xlabel("Dose (mJ/cm²)")
    ax.set_ylabel("log(N₀/Nₜ)")
    ax.set_title("Dose-Response Curve")
    st.pyplot(fig)

# ------------------------
# 执行预测
# ------------------------
if predict_button:
    model = load_model()
    base = get_strain_features(STRAIN_MAPPING[strain])

    # 构建输入数据（只保留模型所需特征）
    input_data = base.copy()
    input_data["Wavelength(nm)"] = wavelength
    input_data["Dose(mJ/cm2)"] = dose
    input_data["Temp"] = temp
    input_data["Ph"] = ph
    input_data["Log N0"] = 6.0

    input_df = pd.DataFrame([input_data])

    # 获取模型特征并对齐
    try:
        model_features = model.feature_names_
    except AttributeError:
        model_features = FEATURE_COLS

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df.reindex(columns=model_features)

    # 显示输入参数
    st.subheader("📋 Input Features")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

    # 执行预测
    try:
        y_pred = model.predict(input_df)[0]
        st.success(f"✅ Predicted Inactivation (log N₀/Nₜ): **{y_pred:.2f}**")
        st.markdown("ℹ️ log(N₀/Nₜ) indicates log reduction. Higher values mean better inactivation.")

        # 响应曲线
        st.subheader("📈 Dose-Response Curve")
        plot_dose_response(model, input_data)

    except Exception as e:
        st.error("❌ Prediction failed. Please check input data or model compatibility.")
        st.exception(e)

else:
    st.info("⬅️ Enter parameters on the left and click the button to predict.")

# ------------------------
# 页面跳转
# ------------------------
st.page_link("pages/custom_input_page.py", label="🔧 自定义菌种预测", icon="🧬")
