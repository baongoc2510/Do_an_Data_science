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




menu = ["Giới thiệu", "Xây dựng mô hình", "Dự đoán giá xe","Danh sách xe giá bất thường"]
choice = st.sidebar.selectbox('Menu', menu)    
if choice == 'Giới thiệu':
    st.markdown("### **ỨNG DỤNG DỰ ĐOÁN GIÁ XE MÁY CŨ VÀ PHÁT HIỆN GIÁ BẤT THƯỜNG**")
    st.image("xe_may_cu.jpg") 
    # --- PHẦN 1: DỰ ĐOÁN GIÁ XE ---
    st.markdown("### **DỰ ĐOÁN GIÁ XE**")
    st.markdown('<div class="bullet">• Ứng dụng cung cấp công cụ hỗ trợ định giá và gợi ý, giúp minh bạch hoá thị trường xe máy cũ và tăng tỉ lệ giao dịch thành công.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Hỗ trợ người bán định giá hợp lý cho xe máy cũ dựa trên các đặc điểm như thương hiệu, năm sản xuất, tình trạng và khu vực.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Giúp người mua so sánh và nhận diện mức giá hợp lý, tránh bị định giá quá cao.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Tối ưu hóa doanh thu và trải nghiệm người dùng cho Chợ Tốt thông qua việc gợi ý mức giá phù hợp, tăng khả năng giao dịch thành công.</div>', unsafe_allow_html=True)

    # --- KHOẢNG CÁCH GIỮA HAI PHẦN ---
    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- PHẦN 2: DANH SÁCH XE GIÁ BẤT THƯỜNG ---
    st.markdown("### **DANH SÁCH XE GIÁ BẤT THƯỜNG**")
    st.markdown('<div class="bullet">• Giúp hệ thống nhanh chóng phát hiện những tin đăng có mức giá chênh lệch đáng kể so với mặt bằng chung của thị trường.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Hỗ trợ sàn giao dịch nhận diện các trường hợp định giá quá thấp (nguy cơ lừa đảo) hoặc quá cao (đặt giá sai lệch).</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Cho phép đội kiểm duyệt tập trung kiểm tra các tin đăng đáng nghi trước, tiết kiệm thời gian và nâng cao hiệu quả xử lý.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Góp phần đảm bảo tính minh bạch, giúp người mua yên tâm hơn khi lựa chọn xe và hạn chế các tin gây nhiễu trên sàn.</div>', unsafe_allow_html=True)
    st.markdown('<div class="bullet">• Bảo vệ người bán uy tín khỏi việc bị cạnh tranh không lành mạnh bởi các tin đăng đặt giá bất hợp lý.</div>', unsafe_allow_html=True)   
    # --- KHOẢNG CÁCH GIỮA HAI PHẦN ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # THÀNH VIÊN ---
    st.markdown("### **THÀNH VIÊN**")
    # dữ liệu
    data = {
        "STT": [1, 2, 3],
        "Họ tên": ["Mai Bảo Ngọc", "Bùi Ngọc Toản", "Nguyễn Vũ Duy"],
        "Vai trò": ["Xây dựng mô hình dự báo giá", "Xây dựng mô hình phát hiện bất thường", "Lập danh sách xe giá bất thường"]
    }
    df = pd.DataFrame(data)

    # hiển thị
    st.table(df.set_index("STT"))  
    
elif choice == 'Xây dựng mô hình':
    st.markdown("### **1. Tiền xử lý dữ liệu**")

    st.markdown("""
    Bộ dữ liệu xe máy cũ được thu thập từ nền tảng *Chợ Tốt*, bao gồm các thuộc tính phản ánh đặc điểm kỹ thuật, mức độ sử dụng và giá rao bán của xe. 
    Trước khi đưa vào mô hình dự báo, dữ liệu được xử lý và chuẩn hóa theo quy trình sau:
    """)

    st.markdown("""
    <ul style="line-height: 1.8;">
    <li>Chuẩn hóa các trường giá (<i>Giá</i>, <i>Khoảng giá min</i>, <i>Khoảng giá max</i>) nhằm đảm bảo tính nhất quán khi phân tích.</li>
    <li>Loại bỏ các bản ghi thiếu dữ liệu quan trọng hoặc chứa giá trị ngoại lai gây ảnh hưởng đến chất lượng mô hình.</li>
    <li>Chuẩn hóa kiểu dữ liệu cho các biến số như <i>Năm đăng ký</i>, <i>Số Km đã đi</i>, … để đảm bảo tương thích với các thuật toán học máy.</li>
    <li>Thực hiện scaling cho các biến liên tục nhằm giảm sai lệch thang đo và cải thiện độ ổn định trong quá trình huấn luyện.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown(""" 
    Các biến phân loại (<i>Thương hiệu</i>, <i>Dòng xe</i>, <i>...</i>) được xử lý bằng <b>StringIndexer</b> và <b>OneHotEncoder</b>. 
    Sau đó toàn bộ đặc trưng được hợp nhất thành một vector đầu vào duy nhất thông qua <b>VectorAssembler</b>.
    """, unsafe_allow_html=True)

    st.markdown("""
    Dữ liệu sau khi chuẩn hóa được chia theo tỷ lệ:
    """)

    st.markdown("""
    <ul style="line-height: 1.8;">
    <li><b>80%</b> dùng để huấn luyện mô hình.</li>
    <li><b>20%</b> dùng để đánh giá hiệu suất dự báo.</li>
    </ul>
    """, unsafe_allow_html=True)




# --- XÂY DỰNG MÔ HÌNH DỰ BÁO GIÁ ---

    st.markdown("### **2. Xây dựng mô hình dự báo giá**")

    st.markdown("""
    Nhóm tiến hành huấn luyện nhiều thuật toán khác nhau nhằm so sánh hiệu năng và lựa chọn mô hình tối ưu, bao gồm:
    """)

    # Bullet list các thuật toán
    st.markdown("""
    <ul style="line-height: 1.8;">
    <li>Linear Regression</li>
    <li>Random Forest Regressor</li>
    <li>Gradient Boosted Trees (GBT)</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    Tất cả các mô hình đều được đánh giá bằng cùng một bộ thước đo:
    """)

    # Bullet list các chỉ số đánh giá
    st.markdown("""
    <ul style="line-height: 1.8;">
    <li><b>MAE (Mean Absolute Error)</b>: sai số dự báo trung bình tuyệt đối giữa giá trị thực tế và giá trị dự đoán.</li>
    <li><b>R² (hệ số xác định)</b>: độ phù hợp của mô hình (càng cao càng tốt).</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("Kết quả huấn luyện mô hình:")

    # --- BẢNG KẾT QUẢ ---
    import pandas as pd

    results = {
        "Mô hình": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosted Trees (GBT)"
        ],
        "MAE (VND)": [
            "6.700.876.000",
            "5.744.014",
            "7.142.370",
        ],
        "R²": [
            "-5,98e+19",
            "0,7518",
            "0,6962",
        ],
        "Nhận xét": [
            "Sai số cực lớn và R² âm nên mô hình hoàn toàn không phù hợp với dữ liệu",
            "Sai số thấp nhất và R² cao nhất, là mô hình cho hiệu suất tốt nhất",
            "Sai số và R² ở mức khá, nhưng vẫn kém hơn Random Forest",
        ]
    }

    df_result = pd.DataFrame(results)

    # Ẩn index
    st.dataframe(df_result, hide_index=True)
    
    st.markdown("""
    Kết quả so sánh mô hình cho thấy Random Forest hoạt động tốt nhất trong ba mô hình, với giá trị MAE ≈ 5.74 và R² ≈ 0.75, cho thấy mô hình giải thích được khoảng 75% phương sai của dữ liệu giá xe và có sai số dự đoán trung bình thấp nhất. Mô hình Gradient Boosting đứng thứ hai, có độ chính xác khá tốt nhưng kém hơn một chút so với Random Forest (MAE ≈ 7.14, R² ≈ 0.70). Ngược lại, Linear Regression cho kết quả rất kém, với MAE cực lớn, R² âm (≈ –5.98e+19), chứng tỏ mô hình tuyến tính không phù hợp với tập dữ liệu này – có thể do mối quan hệ giữa các biến độc lập và giá xe là phi tuyến tính và phức tạp. Như vậy, Random Forest là lựa chọn tối ưu để dự đoán giá xe máy trong trường hợp này.
    """)
    
    st.markdown("### **3. Phát hiện xe máy giá bất thường**")

    st.markdown("""
    Quy trình kiểm tra một mức giá có bất thường hay không được thực hiện dựa trên mô hình dự đoán và thống kê theo từng dòng xe. 
    Hệ thống vận hành theo các bước sau:

    #### **Bước 1 — Nhập giá thực tế từ người dùng**
    Người dùng cung cấp mức giá rao bán để hệ thống so sánh với giá dự đoán và dữ liệu tham chiếu.

    #### **Bước 2 — Kiểm tra điều kiện trước khi đánh giá**
    Hệ thống yêu cầu phải có giá dự đoán của xe (từ mô hình dự báo) trước khi tiến hành kiểm tra.

    #### **Bước 3 — Tính sai số dự báo (Residual)**
    Sai số được tính bằng chênh lệch giữa giá thực và giá dự đoán:
    
    **residual = Giá_thực − Giá_dự_đoán**

    #### **Bước 4 — Lấy giá trị tham chiếu theo dòng xe**
    Hệ thống sử dụng bảng thống kê residual theo từng dòng xe để lấy:
    - mean residual (mean_ref)
    - độ lệch chuẩn residual (std_ref)

    Nếu dòng xe không có dữ liệu, hệ thống dùng trung bình toàn bộ tập dữ liệu.

    #### **Bước 5 — Chuẩn hoá sai số (Residual-z)**
    Sai số được chuẩn hoá để đánh giá mức độ lệch so với thị trường của phân khúc:

    **residual_z = (residual − mean_ref) / std_ref**

    Giá trị này giúp xác định mức giá có lệch bất thường so với nhóm xe tương đồng hay không.

    #### **Bước 6 — Đánh giá bất thường**
    Dựa trên ngưỡng chuẩn hoá:
    - **residual_z > +2** → Giá **đắt bất thường**
    - **residual_z < −2** → Giá **rẻ bất thường**
    - **|residual_z| ≤ 2** → Giá **bình thường**

    Kết quả giúp người dùng và hệ thống nhận diện các tin đăng rao bán lệch so với mặt bằng chung của thị trường.
    """, unsafe_allow_html=True)


    st.markdown("## **4. Lập danh sách tổng hợp các xe có giá bất thường**")

    st.markdown("""
    Bên cạnh việc kiểm tra giá cho từng xe theo yêu cầu của người dùng, hệ thống còn cung cấp chức năng **liệt kê toàn bộ các tin đăng có mức giá bất thường** nhằm hỗ trợ công tác kiểm duyệt của quản trị viên. 
    Mục tiêu của tính năng này là giúp admin nhanh chóng phát hiện những tin rao bán lệch khỏi mặt bằng thị trường và đảm bảo chất lượng dữ liệu trên sàn giao dịch.
    """)

    st.markdown("### **Thông tin hiển thị trong danh sách**")
    st.markdown("""
    Mỗi xe trong danh sách bất thường được trình bày kèm theo:
    <ul style="line-height:1.7;">
    <li><b>Giá thực tế</b> và <b>giá dự đoán</b> từ mô hình.</li>
    <li><b>Residual</b> (mức chênh lệch tuyệt đối).</li>
    <li><b>Residual-z</b>, thể hiện mức độ bất thường theo đơn vị độ lệch chuẩn.</li>
    <li>Thông tin mô tả xe: thương hiệu, dòng xe, loại xe,… để admin đối chiếu nhanh.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("### **Quy trình xử lý của admin**")
    st.markdown("""
    - Admin có thể xem chi tiết từng tin đăng, kiểm tra mô tả và hình ảnh, sau đó đưa ra quyết định: phê duyệt, xác minh lại hoặc từ chối.  
    - Danh sách cung cấp nút tải xuống CSV để phục vụ công tác kiểm tra hàng loạt và lưu trữ hồ sơ kiểm duyệt.
    """)

    st.markdown("### **Lợi ích**")
    st.markdown("""
    <ul style="line-height:1.7;">
    <li>Ngăn chặn các tin rao có giá quá thấp hoặc quá cao một cách bất hợp lý, giảm nhiễu thị trường.</li>
    <li>Hỗ trợ phát hiện sớm các tin có dấu hiệu gian lận hoặc thiếu minh bạch.</li>
    <li>Bảo vệ người mua bằng cách cảnh báo các mức giá không phù hợp.</li>
    <li>Giúp đội kiểm duyệt làm việc hiệu quả hơn, duy trì chất lượng và tính minh bạch của sàn giao dịch.</li>
    </ul>
    """, unsafe_allow_html=True)
    





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

    st.write("### Danh sách các xe bất thường trong dữ liệu")

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