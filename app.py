from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style="background-color:#f0f4f8;padding:20px;border-radius:10px;
                margin-bottom:25px;border-left:6px solid #1f77b4;">
        <h1 style="color:#102a43;margin-bottom:5px;">
            💻 Laptop Price Prediction System
        </h1>
        <p style="color:#486581;font-size:16px;margin:0;">
            Estimate laptop prices in Indian Rupees (₹) using brand and hardware
            specifications with our trained Random Forest pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load the fitted preprocessing + Random Forest pipeline
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    # Check both script directory and current working directory
    script_dir = Path(__file__).resolve().parent
    cwd_dir = Path.cwd()

    possible_paths = [
        script_dir / "models" / "laptop_price_model.joblib",
        cwd_dir / "models" / "laptop_price_model.joblib",
        script_dir.parent / "models" / "laptop_price_model.joblib",
        cwd_dir.parent / "models" / "laptop_price_model.joblib",
        Path("models/laptop_price_model.joblib").resolve(),
    ]

    # Remove duplicates preserving order
    seen = set()
    unique_paths = []
    for p in possible_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    for model_path in unique_paths:
        if model_path.exists() and model_path.is_file():
            return joblib.load(model_path)

    searched_paths = "\n".join(str(path) for path in unique_paths)
    raise FileNotFoundError(
        f"Could not find laptop_price_model.joblib.\nChecked paths:\n{searched_paths}"
    )


try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as error:
    pipeline = None
    model_loaded = False
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Ensure 'laptop_price_model.joblib' is inside the 'models' folder."
    )


# --------------------------------------------------
# Input features
# --------------------------------------------------
st.subheader("🛠️ Configure Laptop Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🏢 Brand & Category")

    company_options = [
        "Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu",
        "Google", "HP", "Huawei", "LG", "Lenovo", "MSI",
        "Mediacom", "Microsoft", "Razer", "Samsung", "Toshiba",
        "Vero", "Xiaomi"
    ]
    company = st.selectbox("Manufacturer / Brand", company_options, index=company_options.index("Dell"))

    type_options = [
        "2 in 1 Convertible", "Gaming", "Netbook",
        "Notebook", "Ultrabook", "Workstation"
    ]
    type_name = st.selectbox("Form Factor / Type", type_options, index=type_options.index("Notebook"))

    opsys_options = [
        "Windows 10", "No OS", "Linux", "Windows 7",
        "Chrome OS", "macOS", "Mac OS X", "Windows 10 S", "Android"
    ]
    op_sys = st.selectbox("Operating System", opsys_options, index=0)

    weight_kg = st.slider("Weight (kg)", min_value=0.6, max_value=4.8, value=2.0, step=0.05)


with col2:
    st.markdown("### ⚡ Processor & Graphics")

    cpu_options = [
        "Intel Core i7", "Intel Core i5", "Intel Core i3",
        "Intel Celeron", "Intel Pentium", "Intel Core M",
        "Intel Xeon", "Intel Atom", "AMD A-Series",
        "AMD Ryzen", "AMD E-Series", "AMD FX", "Samsung Cortex"
    ]
    cpu_family = st.selectbox("CPU Family", cpu_options, index=cpu_options.index("Intel Core i5"))

    clock_speed = st.slider("CPU Clock Speed (GHz)", min_value=0.9, max_value=3.6, value=2.5, step=0.1)

    ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
    ram_gb = st.selectbox("RAM (GB)", ram_options, index=ram_options.index(8))

    gpu_options = [
        "Intel HD", "Nvidia GeForce GTX", "AMD Radeon",
        "Nvidia GeForce Other/MX", "Intel UHD", "Nvidia Quadro",
        "Intel Iris", "AMD FirePro", "ARM Mali", "Intel Other"
    ]
    gpu_category = st.selectbox("Graphics Category (GPU)", gpu_options, index=gpu_options.index("Intel HD"))


with col3:
    st.markdown("### 🖥️ Display & Storage")

    inches = st.selectbox("Screen Size (Inches)", [10.1, 11.6, 12.0, 12.5, 13.3, 14.0, 15.6, 17.3, 18.4], index=6)

    resolution_options = {
        "1920x1080 (Full HD)": (1920, 1080),
        "1366x768 (Standard HD)": (1366, 768),
        "1600x900 (HD+)": (1600, 900),
        "2560x1440 (Quad HD / 2K)": (2560, 1440),
        "2880x1800 (Retina Display)": (2880, 1800),
        "3200x1800 (QHD+)": (3200, 1800),
        "3840x2160 (4K Ultra HD)": (3840, 2160)
    }
    res_label = st.selectbox("Screen Resolution", list(resolution_options.keys()), index=0)
    x_res, y_res = resolution_options[res_label]

    ppi = np.sqrt(x_res**2 + y_res**2) / float(inches)
    st.caption(f"Calculated Pixel Density: **{ppi:.1f} PPI**")

    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        touchscreen = 1 if st.checkbox("Touchscreen", value=False) else 0
    with disp_col2:
        ips = 1 if st.checkbox("IPS Display", value=False) else 0

    ssd_options = [0, 128, 256, 512, 768, 1024, 2048]
    ssd_gb = st.selectbox("Primary Storage: SSD (GB)", ssd_options, index=ssd_options.index(256))

    hdd_options = [0, 500, 1000, 2000]
    hdd_gb = st.selectbox("Secondary Storage: HDD (GB)", hdd_options, index=0)

    flash_gb = 0
    hybrid_gb = 0


st.markdown("---")

# --------------------------------------------------
# Prediction Action
# --------------------------------------------------
predict_btn = st.button("🚀 Estimate Laptop Price", type="primary", use_container_width=True)

if predict_btn:
    if not model_loaded:
        st.error("Model is not loaded. Please verify the model file exists in the 'models' directory.")
    else:
        input_data = pd.DataFrame([{
            "Company": company,
            "TypeName": type_name,
            "Inches": float(inches),
            "Ram_GB": int(ram_gb),
            "Weight_kg": float(weight_kg),
            "Touchscreen": int(touchscreen),
            "Ips": int(ips),
            "Ppi": float(ppi),
            "Cpu_Family": cpu_family,
            "ClockSpeed": float(clock_speed),
            "Gpu_Category": gpu_category,
            "SSD_GB": int(ssd_gb),
            "HDD_GB": int(hdd_gb),
            "Flash_GB": int(flash_gb),
            "Hybrid_GB": int(hybrid_gb),
            "OpSys": op_sys
        }])

        try:
            raw_prediction = pipeline.predict(input_data)[0]

            # In case your model was trained on log(Price), handle exp transform automatically:
            # (If raw_prediction is small like 9.x - 11.x, it's log-transformed)
            if raw_prediction < 20.0:
                predicted_price = float(np.exp(raw_prediction))
            else:
                predicted_price = float(raw_prediction)

            predicted_price = max(9000.0, predicted_price)

            st.markdown(
                f"""
                <div style="background-color:#ebfbee;border:2px solid #38d9a9;border-radius:12px;
                            padding:25px;text-align:center;margin-top:15px;">
                    <h3 style="color:#2b8a3e;margin-bottom:5px;">Estimated Retail Price</h3>
                    <h1 style="color:#2b8a3e;font-size:42px;margin:0;">₹ {predicted_price:,.2f}</h1>
                    <p style="color:#555;font-size:14px;margin-top:8px;">
                        Prediction generated using the trained Random Forest Regressor Pipeline
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("🔍 View Configured Features Sent to Model"):
                st.table(input_data)

        except Exception as err:
            st.error(f"Prediction error: {err}")