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




menu = ["Home", "Giới thiệu", "Dự đoán giá xe", "Tìm kiếm xe"]
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

elif choice=='Tìm kiếm xe':
    DATA_PATH = "data_motobikes.xlsx"
    COSINE_PKL = "cosine_sim_model.pkl"
    UPLOADED_IMAGE = "b12bca47-fea2-499d-80f1-1915896b8525.png"

    # ---------------- Helpers & Caches ----------------
    @st.cache_resource(ttl=3600)
    def load_data(path):
        try:
            df = pd.read_excel(path, engine="openpyxl")
            df = df.reset_index(drop=True)
            return df
        except Exception as e:
            st.error(f"Không thể đọc file dữ liệu: {path}\n{e}")
            return None

    @st.cache_resource(ttl=3600)
    def load_cosine(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                cosine = pickle.load(f)
            return cosine
        except Exception as e:
            st.error(f"Lỗi khi load ma trận cosine từ {path}: {e}")
            return None

    def find_best_title_match(df_titles, query):
        best_idx = None
        best_score = 0.0
        q = str(query).strip().lower()
        if not q:
            return None, 0.0
        for idx, title in enumerate(df_titles):
            t = str(title).lower()
            score = SequenceMatcher(None, q, t).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def get_recommendations_by_index(df, cosine_sim, idx, top_k=30):
        if cosine_sim is None:
            return pd.DataFrame()
        try:
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = [s for s in sim_scores if s[0] != idx]
            top_scores = sim_scores[:top_k]
            indices = [i for i, _ in top_scores]
            return df.iloc[indices].reset_index(drop=True)
        except Exception as e:
            st.error(f"Lỗi khi lấy gợi ý từ ma trận cosine: {e}")
            return pd.DataFrame()

    def display_rows_with_expander(df_rows):
        if df_rows is None or df_rows.empty:
            st.write("_Không có kết quả để hiển thị._")
            return

        c0, c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1, 1])
        c0.markdown("**Tiêu đề**")
        c1.markdown("**Thương hiệu**")
        c2.markdown("**Dòng xe**")
        c3.markdown("**Năm đăng ký**")
        c4.markdown("**Giá**")
        c5.markdown("**Chi tiết**")

        for _, row in df_rows.iterrows():
            t0, t1, t2, t3, t4, t5 = st.columns([3, 2, 2, 1, 1, 1])
            t0.write(row.get("Tiêu đề", ""))
            t1.write(row.get("Thương hiệu", ""))
            t2.write(row.get("Dòng xe", ""))
            t3.write(row.get("Năm đăng ký", ""))
            t4.write(row.get("Giá", ""))
            bike_id = row.get("id", "")
            label = f"Chi tiết ({bike_id})"
            with t5:
                with st.expander(label):
                    desc = row.get("Mô tả chi tiết", "")
                    if desc:
                        st.write(desc)
                    else:
                        st.write("_Không có mô tả chi tiết._")

    def paginate_dataframe(df, page, per_page):
        if df is None:
            return pd.DataFrame()
        start = (page - 1) * per_page
        end = start + per_page
        return df.iloc[start:end].reset_index(drop=True)

    # ---------------- Main ----------------
    def main():
        # header image
        try:
            st.image(UPLOADED_IMAGE, use_container_width=True)
        except Exception:
            pass

        st.title("Tìm kiếm xe")


        # load
        df_bikes = load_data(DATA_PATH)
        cosine_sim = load_cosine(COSINE_PKL)
        if df_bikes is None:
            st.stop()
        if cosine_sim is None:
            st.warning(f"Không tìm thấy hoặc không load được ma trận cosine từ '{COSINE_PKL}'. Chức năng gợi ý sẽ không hoạt động.")

        # session init (safe defaults)
        if 'random_bikes' not in st.session_state:
            st.session_state.random_bikes = df_bikes.head(10).reset_index(drop=True)
        if 'selected_bike_id' not in st.session_state:
            st.session_state.selected_bike_id = None
        if 'page' not in st.session_state:
            st.session_state.page = 1
        if 'max_results' not in st.session_state:
            st.session_state.max_results = 30
        if 'per_page' not in st.session_state:
            st.session_state.per_page = 6
        if 'last_query' not in st.session_state:
            st.session_state.last_query = ""
        if 'last_query_method' not in st.session_state:
            st.session_state.last_query_method = ""

        # function callbacks
        def refresh_random_list():
            try:
                st.session_state.random_bikes = df_bikes.sample(n=10).reset_index(drop=True)
                # reset selectbox key so user must re-select
                st.session_state.selected_bike_id = None
                st.session_state.last_query = ""
                st.session_state.last_query_method = ""
                st.session_state.page = 1
                # clear the selectbox stored value
                st.session_state.pop("selected_bike_option", None)
            except Exception as e:
                st.error("Lỗi khi làm mới danh sách: " + str(e))

        def on_select_change():
            # called when user picks an item from selectbox A
            val = st.session_state.get("selected_bike_option", None)
            if val:
                try:
                    # val is a tuple (title, id)
                    st.session_state.selected_bike_id = val[1]
                    st.session_state.last_query = str(st.session_state.selected_bike_id)
                    st.session_state.last_query_method = "selectbox"
                    st.session_state.page = 1
                except Exception:
                    pass

        # --- Search UI: selection A and typed input B (settings inside) ---
        st.markdown("---")
        colA1, colA2 = st.columns([4, 1])
        with colA1:
            bike_options = [(row['Tiêu đề'], row['id']) for _, row in st.session_state.random_bikes.iterrows()]
            # note: use on_change callback
            st.selectbox(
                "Danh sách",
                options=bike_options,
                format_func=lambda x: x[0] if isinstance(x, tuple) else str(x),
                key="selected_bike_option",
                on_change=on_select_change
            )
        with colA2:
            if st.button("Làm mới danh sách"):
                refresh_random_list()

        q_input = st.text_input("Nhập từ khóa:", value="")

        # settings placed inside search area as requested
        st.markdown("**Thiết lập gợi ý**")
        cols_set = st.columns([1,1,2])
        with cols_set[0]:
            max_results = st.number_input("Số gợi ý tối đa (tổng)", min_value=5, max_value=500,
                                        value=st.session_state.max_results, step=5, key="input_max_results")
        with cols_set[1]:
            per_page = st.selectbox("Số kết quả / trang", options=[3,4,6,10],
                                    index=[3,4,6,10].index(st.session_state.per_page) if st.session_state.per_page in [3,4,6,10] else 2,
                                    key="input_per_page")
    

        # sync to session_state (persist)
        st.session_state.max_results = int(max_results)
        st.session_state.per_page = int(per_page)

        # Button for typed search (B)
        if st.button("Tìm kiếm"):
            if str(q_input).strip() == "":
                st.info("Hãy nhập id hoặc từ khóa vào ô tìm kiếm.")
            else:
                st.session_state.page = 1
                st.session_state.last_query = str(q_input).strip()
                st.session_state.last_query_method = "typed"
                # no explicit rerun required; widget changes cause rerun automatically

        # ------------------ Processing search (if there's a last_query) ------------------
        last_q = st.session_state.get('last_query', "")
        method = st.session_state.get('last_query_method', "")
        if last_q:
            chosen_index = None
            chosen_method = None

            if method == "selectbox":
                # last_q is id
                try:
                    q_num = int(last_q)
                    matches = df_bikes.index[df_bikes["id"] == q_num].tolist()
                    if matches:
                        chosen_index = matches[0]
                        chosen_method = f"id chính xác ({q_num})"
                    else:
                        st.warning(f"Không tìm thấy id = {q_num} trong dữ liệu.")
                except Exception:
                    st.warning("ID chọn không hợp lệ.")
            else:
                # typed: could be id or keyword
                if last_q.isdigit():
                    q_num = int(last_q)
                    matches = df_bikes.index[df_bikes["id"] == q_num].tolist()
                    if matches:
                        chosen_index = matches[0]
                        chosen_method = f"id chính xác ({q_num})"
                if chosen_index is None:
                    best_idx, best_score = find_best_title_match(df_bikes["Tiêu đề"].astype(str).tolist(), last_q)
                    if best_idx is not None and best_score > 0.05:
                        chosen_index = best_idx
                        chosen_method = f"closest title match (score={best_score:.3f})"
                    else:
                        st.warning("Không tìm thấy Tiêu đề nào giống query. Hãy thử từ khóa khác.")
                        chosen_index = None

            # If have chosen index -> use cosine to get recommendations
            if chosen_index is not None:
                st.success(f"Đã chọn item index = {chosen_index} bằng phương pháp: {chosen_method}")

                if cosine_sim is None:
                    st.error("Ma trận cosine chưa sẵn sàng. Không thể tạo gợi ý.")
                else:
                    recommendations = get_recommendations_by_index(df_bikes, cosine_sim, chosen_index,
                                                                top_k=st.session_state.max_results)
                    if recommendations.empty:
                        st.write("_Không có gợi ý_")
                    else:
                        total = len(recommendations)
                        total_pages = max(1, ceil(total / st.session_state.per_page))
                        st.write(f"Tổng gợi ý thu được: **{total}** — Hiển thị **{st.session_state.per_page}** / trang — Tổng trang: **{total_pages}**")

                        # normalize page in session_state
                        if st.session_state.page < 1:
                            st.session_state.page = 1
                        if st.session_state.page > total_pages:
                            st.session_state.page = total_pages

                        # page chooser (persisted)
                        new_page = st.number_input("Chọn trang", min_value=1, max_value=total_pages,
                                                value=st.session_state.page, step=1, key="ui_page")
                        if new_page != st.session_state.page:
                            st.session_state.page = int(new_page)

                        # slice and display
                        df_page = paginate_dataframe(recommendations, st.session_state.page, st.session_state.per_page)
                        display_rows_with_expander(df_page)

                        # navigation buttons (update session_state.page)
                        nav_col1, nav_col2, _ = st.columns([1,1,4])
                        with nav_col1:
                            if st.button("<< Trang trước"):
                                st.session_state.page = max(1, st.session_state.page - 1)
                        with nav_col2:
                            if st.button("Trang sau >>"):
                                st.session_state.page = min(total_pages, st.session_state.page + 1)

        # footer note
        st.markdown("---")
        st.caption("Ghi chú: Đảm bảo ma trận cosine tương ứng đúng thứ tự dòng với dataframe (sử dụng df.reset_index(drop=True) khi tạo ma trận).")

    if __name__ == "__main__":
        main()    