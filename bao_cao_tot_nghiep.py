import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import re
import os
import joblib
from datetime import datetime
from sklearn.ensemble import IsolationForest
from difflib import SequenceMatcher
from math import ceil

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .main {
        padding-right: 0rem !important;
        padding-left: 0rem !important;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)




menu = ["Home", "Giới thiệu", "Dự đoán giá xe", "Danh sách xe giá bất thường"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.title("Trung Tâm Tin Học")
    st.image("xe_may_cu.jpg")    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Giới thiệu':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 2 chủ đề trong khóa học:    
    - Topic 1: Dự đoán giá xe máy cũ, phát hiện xe máy bất thường
    - Topic 2: Hệ thống gợi ý xe máy dựa trên nội dung, phân cụm xe máy
             """)
elif choice == 'Dự đoán giá xe':
    ###### Giao diện Streamlit ######
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
            dong_xe = st.session_state.get("dong_xe", None)

            residual = gia_thuc - gia_du_doan_vnd

            # lấy mean/std từ stats (đã load từ CSV trước đó)
            if dong_xe is not None and dong_xe in stats.index:
                mean_ref = stats.loc[dong_xe, "mean"]
                std_ref  = stats.loc[dong_xe, "std"]
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
elif choice == 'Danh sách xe giá bất thường':

    st.write("### 3. Danh sách các xe bất thường trong dữ liệu")

    # --- 0. Nếu df chưa có (vì bạn có thể chỉ load df ở branch khác), cố gắng load từ file Excel ---
    if 'df' not in globals() and 'df' not in locals():
        try:
            df = pd.read_excel("data_motobikes.xlsx")
            st.info("Đã load dữ liệu từ data_motobikes.xlsx")
        except Exception as e:
            st.error(f"Không tìm thấy DataFrame 'df' và không thể load file data_motobikes.xlsx: {e}")
            st.stop()

    # --- 0.5. Nếu model chưa load, cố gắng load model (cần để dự đoán) ---
    if 'model' not in globals() and 'model' not in locals():
        try:
            model = joblib.load("motobike_price_model_project_1.pkl")
            st.info("Đã load model dự đoán.")
        except Exception as e:
            st.error(f"Không tìm thấy model và không thể load motobike_price_model_project_1.pkl: {e}")
            st.stop()

    # --- 0.75. Nếu stats chưa load, cố gắng load file residual stats ---
    if 'stats' not in globals() and 'stats' not in locals():
        try:
            stats = pd.read_csv("residual_stats_by_group.csv", index_col=0)
            st.info("Đã load residual_stats_by_group.csv")
        except Exception as e:
            st.error(f"Không tìm thấy residual_stats_by_group.csv: {e}")
            st.stop()

    # --- 1. Chuẩn hóa bản sao của df (không sửa df gốc) ---
    df_local = df.copy()
    # Chuyển Giá sang số (loại bỏ ký tự không phải số)
    df_local["Giá"] = df_local["Giá"].astype(str).str.replace(r"[^\d]", "", regex=True)
    df_local["Giá"] = pd.to_numeric(df_local["Giá"], errors="coerce")
    # Năm đăng ký -> numeric, tính Tuổi xe
    df_local["Năm đăng ký"] = pd.to_numeric(df_local["Năm đăng ký"], errors="coerce")
    df_local["Tuổi xe"] = datetime.now().year - df_local["Năm đăng ký"]

    # --- 2. Dự đoán (vectorized nếu được, fallback vòng lặp nếu model không chấp nhận DataFrame) ---
    features = [
        'Thương hiệu','Dòng xe','Tình trạng','Loại xe',
        'Dung tích xe','Năm đăng ký','Tuổi xe','Xuất xứ',
        'Chính sách bảo hành','Số Km đã đi'
    ]

    with st.spinner("Đang dự đoán cho toàn bộ dataset (một lần) ..."):
        try:
            X = df_local[features]
            y_hat = model.predict(X)
            y_hat = np.array(y_hat, dtype=float) * 1_000_000   # giữ logic nhân triệu nếu model trả về triệu
            df_local["Giá dự đoán"] = y_hat
        except Exception:
            # fallback từng dòng
            predicted = []
            for _, r in df_local.iterrows():
                x = pd.DataFrame([{c: r[c] for c in features}])
                try:
                    y = model.predict(x)[0]
                    predicted.append(float(y) * 1_000_000)
                except Exception:
                    predicted.append(np.nan)
            df_local["Giá dự đoán"] = predicted

    # --- 3. Tính residual và join stats theo 'Dòng xe' (hoặc dùng index sẵn có) ---
    df_local["Residual"] = df_local["Giá"] - df_local["Giá dự đoán"]

    if "Dòng xe" in stats.columns:
        stats_idx = stats.set_index("Dòng xe")
    else:
        stats_idx = stats

    df_local = df_local.join(stats_idx, on="Dòng xe", how="left")

    # Tính z-score (cẩn trọng với NaN / std = 0)
    df_local["Residual_z"] = (df_local["Residual"] - df_local["mean"]) / df_local["std"]

    # --- 4. Lọc và hiển thị kết quả ---
    df_abnormal = df_local[(df_local["Residual_z"] > 2) | (df_local["Residual_z"] < -2)].dropna(subset=["Residual_z"])

    if df_abnormal.empty:
        st.success("✔ Không có xe bất thường trong dataset.")
    else:
        st.error(f"💥 Có {len(df_abnormal)} xe bất thường:")
        st.dataframe(
            df_abnormal[["Thương hiệu","Dòng xe","Loại xe","Giá","Giá dự đoán","Residual","Residual_z"]]
            .sort_values("Residual_z", ascending=False)
        )
        csv_bytes = df_abnormal.to_csv(index=False).encode("utf-8")
        st.download_button("Tải toàn bộ danh sách bất thường (.csv)", csv_bytes, file_name="xe_bat_thuong.csv")
