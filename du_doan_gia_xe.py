# app_predict_motobike.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest

###### Giao diện Streamlit ######
st.set_page_config(page_title="Dự đoán giá xe máy", layout="centered")
st.image("xe_may_cu.jpg", use_container_width=True)
st.title("Dự đoán giá xe máy")


# load model dự đoán giá
@st.cache_resource(ttl=3600)
def load_model(path="bmotobike_price_model_project_1.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Không thể load model từ {path}: {e}")
        return None

model = load_model("motobike_price_model_project_1.pkl")  


# đọc dữ liệu từ file data_motobikes.xlsx
df = pd.read_excel("data_motobikes.xlsx")
st.dataframe(df.head())   

# Trường hợp 2: Đọc dữ liệu từ file csv, excel do người dùng tải lên
st.write("### Tải file dữ liệu")

uploaded_file = st.file_uploader(
    "Chọn file dữ liệu (CSV hoặc Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    file_name = uploaded_file.name

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.write("Dữ liệu đã nhập:")
    st.dataframe(df.head())
        
        
st.write("### 1. Dự đoán giá xe máy cũ")
    # Tạo điều khiển để người dùng nhập các thông tin về xe máy

# Chọn hãng xe
thuong_hieu = st.selectbox(
    "Chọn hãng xe",
    df['Thương hiệu'].unique()
)

# Lọc dữ liệu theo hãng vừa chọn
df_filtered = df[df['Thương hiệu'] == thuong_hieu]
# Chọn dòng xe phụ thuộc vào hãng
dong_xe = st.selectbox(
    "Chọn dòng xe",
    df_filtered['Dòng xe'].unique()    
)
tinh_trang = st.selectbox("Chọn tình trạng", df['Tình trạng'].unique())

# Lọc dữ liệu theo dòng xe vừa chọn
df_filtered_by_dong = df_filtered[df_filtered['Dòng xe'] == dong_xe]
# Chọn loại xe phụ thuộc vào dòng xe
loai_xe = st.selectbox(
    "Chọn loại xe",
    df_filtered_by_dong['Loại xe'].unique()
)
dung_tich_xi_lanh = st.selectbox("Dung tích xi lanh (cc)", df['Dung tích xe'].unique())
nam_dang_ky = st.slider("Năm đăng ký", 2000, 2024, 2015)
Tuoi_xe = datetime.now().year - nam_dang_ky
xuat_xu = st.selectbox("Xuất xứ", df['Xuất xứ'].unique())
chinh_sach_bao_hanh = st.selectbox("Chính sách bảo hành", df['Chính sách bảo hành'].unique())
so_Km_da_di = st.number_input("Số Km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
du_doan_gia = st.button("Dự đoán giá")




if du_doan_gia:
    input_data = pd.DataFrame([{
        'Thương hiệu': thuong_hieu,
        'Dòng xe': dong_xe,
        'Tình trạng': tinh_trang,
        'Loại xe': loai_xe,
        'Dung tích xe': dung_tich_xi_lanh,
        'Năm đăng ký': nam_dang_ky,
        'Tuổi xe': Tuoi_xe,
        'Xuất xứ': xuat_xu,
        'Chính sách bảo hành': chinh_sach_bao_hanh,
        'Số Km đã đi': so_Km_da_di
    }])

    # Dự đoán bằng model đã load
    y_pred = model.predict(input_data)

    gia_du_doan = float(y_pred[0])

    # Nếu mô hình của bạn dự đoán theo triệu → đổi ra VND
    gia_du_doan_vnd = int(gia_du_doan * 1_000_000)

    st.success(f"Giá dự đoán: {gia_du_doan_vnd:,.0f} VND")
    st.session_state['gia_du_doan_vnd'] = gia_du_doan_vnd





st.write("### 2. Phát hiện xe máy giá bất thường")
stats = pd.read_csv("residual_stats_by_group.csv", index_col=0)



gia_thuc = st.number_input(
    "Nhập giá thực (VND):",
    min_value=0,
    value=15_000_000,
    step=100_000,
    key="gia_thuc_input"
)

# nút để người dùng chủ động yêu cầu kiểm tra
kiem_tra = st.button("Kiểm tra bất thường")

# chỉ khi bấm nút mới tính và hiển thị kết quả
if kiem_tra:
    if "gia_du_doan_vnd" not in st.session_state:
        st.info("Hãy bấm 'Dự đoán giá' trước để có giá dự đoán.")
    else:
        gia_du_doan_vnd = st.session_state["gia_du_doan_vnd"]
        loai_xe = st.session_state.get("loai_xe", None)

        residual = gia_thuc - gia_du_doan_vnd

        # lấy mean/std từ stats (đã load từ CSV trước đó)
        if loai_xe is not None and loai_xe in stats.index:
            mean_ref = stats.loc[loai_xe, "mean"]
            std_ref  = stats.loc[loai_xe, "std"]
        else:
            mean_ref = stats["mean"].mean()
            std_ref  = stats["std"].mean()

        if pd.isna(std_ref) or std_ref == 0:
            st.warning("Không đủ dữ liệu tham chiếu (std = 0).")
        else:
            residual_z = (residual - mean_ref) / std_ref
            if residual_z > 2:
                st.error("💥 ĐẮT BẤT THƯỜNG")
            elif residual_z < -2:
                st.error("💥 RẺ BẤT THƯỜNG")
            else:
                st.success("✔ Bình thường")