import streamlit as st
import joblib
import numpy as np
import pandas as pd


# 加载保存的随机森林模型
model = joblib.load('XGB.pkl')

# 加载模型时添加异常捕获
try:
    model = joblib.load('XGB.pkl')
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Vw": {"type": "numerical", "min": 0, "max": 1000000, "default": 10400.0},
    "Bave": {"type": "numerical", "min": 0, "max": 100, "default": 50.0},
    "hd": {"type": "numerical", "min": 0, "max": 100, "default": 50},
    "hb": {"type": "numerical", "min": 0, "max": 100, "default": 50.0},
}

# Streamlit 界面
st.title("Qp Prediction with XGBoost")

# 动态生成输入项
st.header("Input Feature Values")
feature_values = []
for feature, props in feature_ranges.items():
    value = st.number_input(
        f"{feature} ({props['min']} - {props['max']})",
        min_value=float(props["min"]),
        max_value=float(props["max"]),
        value=float(props["default"]),
    )
    feature_values.append(value)

# 预测逻辑
if st.button("Predict Qp"):
    try:
        # 转换为模型输入格式
        input_data = pd.DataFrame([feature_values], columns=feature_ranges.keys())
        prediction = model.predict(input_data)[0]

        # 显示结果
        st.markdown(f"**Predicted Qp:** `{prediction:.2f}`")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

    '''# 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")'''
