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
    
    # Hiển thị mô tả quy trình phát hiện giá bất thường (chỉ hiển thị văn bản)
    st.markdown("### **3. Quy trình phát hiện giá bất thường**")

    st.markdown("""
    **Bước 1 — Tính sai số dự báo (Residual)**  
    `residual = giá thực − giá dự đoán` — cho biết giá đang rẻ hơn hay đắt hơn so với dự đoán.

    **Bước 2 — Lấy thống kê phân khúc**  
    Hệ thống lấy mean/std, khoảng hợp lý (min–max) và phân vị (P10–P90) theo dòng xe; nếu không có, dùng thống kê toàn bộ.

    **Bước 3 — Chuẩn hoá (Residual-z)**  
    Tính `residual_z = (residual − mean) / std` để đánh giá mức lệch theo chuẩn.

    **Bước 4 — Kiểm tra vi phạm**  
    - Kiểm tra **min–max** (ngoài khoảng hợp lý)  
    - Kiểm tra **P10–P90** (ngoài vùng phổ biến)

    **Bước 5 — Tính điểm bất thường (0–100)**  
    Kết hợp 3 tín hiệu với trọng số: **40% residual-z**, **40% min/max**, **20% P10–P90**.

    **Bước 6 — Phân loại**  
    Theo thứ tự ưu tiên:  
    1. Vi phạm min/max → bất thường (rẻ/đắt)  
    2. Vi phạm P10–P90 → bất thường  
    3. `|residual_z| ≥ 2` → bất thường  
    4. `anomaly_score ≥ 45` → bất thường  
    Nếu không rơi vào các trường hợp trên → **BÌNH THƯỜNG**.

    **Bước 7 — Kết luận hiển thị**  
    Chỉ hiện 1 trong 3 nhãn cho người dùng: **ĐẮT BẤT THƯỜNG**, **RẺ BẤT THƯỜNG**, **BÌNH THƯỜNG**.
    """, unsafe_allow_html=True)


    st.markdown("## **4. Lập danh sách tổng hợp các xe có giá bất thường**")

    st.markdown("""
    Bên cạnh việc kiểm tra giá cho từng xe theo yêu cầu của người dùng, hệ thống còn cung cấp chức năng **liệt kê toàn bộ các tin đăng có mức giá bất thường** nhằm hỗ trợ công tác kiểm duyệt của quản trị viên. 
    Mục tiêu của tính năng này là giúp admin nhanh chóng phát hiện những tin rao bán lệch khỏi mặt bằng thị trường và đảm bảo chất lượng dữ liệu trên sàn giao dịch.
    """)

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
    import joblib
    import numpy as np
    from datetime import datetime
    st.title("🔍 Dự đoán giá xe máy & Phát hiện giá bất thường")
    st.image("xe_may_cu.jpg", use_container_width=True)

    # -------------------------
    # 1) Load model dự đoán
    # -------------------------
    @st.cache_resource(ttl=3600)
    def load_price_model(path="motobike_price_model_project_1.pkl"):
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Không thể load model dự đoán từ {path}: {e}")
            return None

    model = load_price_model("motobike_price_model_project_1.pkl")

    # -------------------------
    # 2) Load dữ liệu mẫu & stats phân khúc
    # -------------------------
    # (Nếu bạn đã load df ở scope ngoài, có thể bỏ phần này và dùng df có sẵn)
    try:
        df = pd.read_excel("data_motobikes.xlsx")
    except Exception:
        st.error("Không thể đọc file data_motobikes.xlsx. Kiểm tra file hoặc đường dẫn.")
        df = pd.DataFrame()
    try:
        stats = pd.read_csv("residual_stats_by_group.csv", index_col=0)
    except Exception:
        st.warning("Không tìm thấy residual_stats_by_group.csv; sử dụng thống kê toàn bộ dataset làm fallback.")
        stats = None

    # -------------------------
    # 3) Form nhập thông tin xe
    # -------------------------
    st.write("## 1. Nhập thông tin xe để dự đoán giá")

    # defensive: nếu df rỗng thì show input tối giản
    if df.empty:
        thuong_hieu = st.text_input("Chọn/nhập hãng xe")
        dong_xe = st.text_input("Chọn/nhập dòng xe")
        tinh_trang = st.text_input("Tình trạng")
        loai_xe = st.text_input("Loại xe")
        dung_tich = st.text_input("Dung tích xe (cc)")
        nam_dang_ky = st.number_input("Năm đăng ký", min_value=1900, max_value=datetime.now().year, value=2015)
        tuoi_xe = datetime.now().year - nam_dang_ky
        xuat_xu = st.text_input("Xuất xứ")
        cs_bh = st.text_input("Chính sách bảo hành")
        so_km = st.number_input("Số Km đã đi", min_value=0, max_value=200000, value=50000, step=1000)
    else:
        thuong_hieu = st.selectbox("Chọn hãng xe", df['Thương hiệu'].unique())
        df_brand = df[df['Thương hiệu'] == thuong_hieu]
        dong_xe = st.selectbox("Chọn dòng xe", df_brand['Dòng xe'].unique())
        df_dong = df_brand[df_brand['Dòng xe'] == dong_xe]
        tinh_trang = st.selectbox("Tình trạng xe", df['Tình trạng'].unique())
        loai_xe = st.selectbox("Loại xe", df_dong['Loại xe'].unique())
        dung_tich = st.selectbox("Dung tích xe (cc)", df['Dung tích xe'].unique())
        nam_dang_ky = st.slider("Năm đăng ký", 2000, datetime.now().year, 2015)
        tuoi_xe = datetime.now().year - nam_dang_ky
        xuat_xu = st.selectbox("Xuất xứ", df['Xuất xứ'].unique())
        cs_bh = st.selectbox("Chính sách bảo hành", df['Chính sách bảo hành'].unique())
        so_km = st.number_input("Số Km đã đi", min_value=0, max_value=200000, value=50000, step=1000)

    # nút dự đoán
    if st.button("💡 Dự đoán giá"):
        if model is None:
            st.error("Model dự đoán chưa load được. Kiểm tra file model.")
        else:
            input_data = pd.DataFrame([{
                "Thương hiệu": thuong_hieu,
                "Dòng xe": dong_xe,
                "Tình trạng": tinh_trang,
                "Loại xe": loai_xe,
                "Dung tích xe": dung_tich,
                "Năm đăng ký": nam_dang_ky,
                "Tuổi xe": tuoi_xe,
                "Xuất xứ": xuat_xu,
                "Chính sách bảo hành": cs_bh,
                "Số Km đã đi": so_km
            }])

            # dự đoán (bắt lỗi nếu model không chấp nhận input)
            try:
                y_pred = model.predict(input_data)
                # nếu model trả về theo "triệu" thì nhân 1e6
                gia_du_doan_vnd = int(float(y_pred[0]) * 1_000_000)
            except Exception as e:
                st.error(f"Không thể dự đoán: {e}")
                gia_du_doan_vnd = None

            if gia_du_doan_vnd is not None:
                st.session_state["gia_du_doan_vnd"] = gia_du_doan_vnd
                st.session_state["dong_xe"] = dong_xe
                st.session_state["input_row"] = input_data.iloc[0].to_dict()
                st.success(f"💰 Giá dự đoán: {gia_du_doan_vnd:,.0f} VND")

    # -------------------------
    # 4) Kiểm tra bất thường  
    # -------------------------
    st.write("## 2. Đánh giá giá bất thường")

    gia_thuc = st.number_input("Nhập giá muốn bán (VND)", min_value=0, value=15_000_000, step=100_000)

    if st.button("📌 Đánh giá"):
        if "gia_du_doan_vnd" not in st.session_state:
            st.warning("Hãy bấm 'Dự đoán giá' trước để có giá dự đoán.")
        else:
            gia_du_doan_vnd = st.session_state["gia_du_doan_vnd"]
            dong_xe_sel = st.session_state.get("dong_xe", None)

            # -------------------------
            # Bước 1: residual
            # -------------------------
            resid = gia_thuc - gia_du_doan_vnd

            # -------------------------
            # Bước 2: lấy thống kê phân khúc (mean, std, min, max, p10, p90)
            # -------------------------
            if stats is not None and pd.notna(dong_xe_sel) and dong_xe_sel in stats.index:
                seg = stats.loc[dong_xe_sel]
                mean_ref = seg.get("mean", stats["mean"].mean())
                std_ref  = seg.get("std", stats["std"].mean())
                seg_min  = seg.get("min", np.nan)
                seg_max  = seg.get("max", np.nan)
                p10      = seg.get("p10", np.nan)
                p90      = seg.get("p90", np.nan)
            else:
                if "Giá" in df.columns and not df["Giá"].isna().all():
                    mean_ref = df["Giá"].mean()
                    std_ref  = df["Giá"].std()
                    seg_min  = df["Giá"].min()
                    seg_max  = df["Giá"].max()
                    p10      = df["Giá"].quantile(0.10)
                    p90      = df["Giá"].quantile(0.90)
                else:
                    mean_ref = 0.0
                    std_ref  = 1.0
                    seg_min  = np.nan
                    seg_max  = np.nan
                    p10 = p90 = np.nan

            # -------------------------
            # Bước 3: Residual-z
            # -------------------------
            if pd.isna(std_ref) or std_ref == 0:
                residual_z = np.nan
            else:
                residual_z = (resid - mean_ref) / std_ref

            # -------------------------
            # Bước 4: Min/Max và P10–P90 violations
            # -------------------------
            minmax_violation = 1 if (
                (not pd.isna(seg_min) and gia_thuc < seg_min) or 
                (not pd.isna(seg_max) and gia_thuc > seg_max)
            ) else 0

            percentile_violation = 1 if (
                (not pd.isna(p10) and gia_thuc < p10) or 
                (not pd.isna(p90) and gia_thuc > p90)
            ) else 0

            # -------------------------
            # Bước 5: Tính điểm bất thường tổng (nội bộ)
            # -------------------------
            w1, w2, w3 = 0.40, 0.40, 0.20
            cap_z = 5.0

            # residual-z score: chuẩn hoá 0–100 (nếu residual_z là NaN -> 0)
            residual_score = min(1.0, abs(residual_z) / cap_z) * 100 if pd.notna(residual_z) else 0.0
            minmax_score = minmax_violation * 100
            p10p90_score = percentile_violation * 100

            anomaly_score = (
                w1 * residual_score +
                w2 * minmax_score +
                w3 * p10p90_score
            )
            anomaly_score = float(np.clip(anomaly_score, 0, 100))

            # -------------------------
            # Bước 6: PHÂN LOẠI
            # - Nếu vi phạm min/max hoặc p10/p90 ⇒ BẤT THƯỜNG NGAY (theo hướng)
            # - Else nếu residual_z có giá trị và |residual_z| >= 2 ⇒ BẤT THƯỜNG (theo hướng)
            # - Else nếu anomaly_score >= 45 ⇒ BẤT THƯỜNG (theo hướng)
            # - Else ⇒ BÌNH THƯỜNG
            # -------------------------
            label = "BÌNH THƯỜNG"

            # 1) ưu tiên vi phạm khoảng (min/max)
            if minmax_violation:
                if pd.notna(seg_min) and gia_thuc < seg_min:
                    label = "RẺ BẤT THƯỜNG"
                elif pd.notna(seg_max) and gia_thuc > seg_max:
                    label = "ĐẮT BẤT THƯỜNG"
                else:
                    label = "BÌNH THƯỜNG"

            # 2) nếu không bị minmax, kiểm tra phân vị
            elif percentile_violation:
                if pd.notna(p10) and gia_thuc < p10:
                    label = "RẺ BẤT THƯỜNG"
                elif pd.notna(p90) and gia_thuc > p90:
                    label = "ĐẮT BẤT THƯỜNG"
                else:
                    label = "BÌNH THƯỜNG"

            # 3) nếu không có vi phạm phân khúc, dùng residual_z nếu khả dụng
            elif pd.notna(residual_z) and abs(residual_z) >= 2.0:
                label = "ĐẮT BẤT THƯỜNG" if resid > 0 else "RẺ BẤT THƯỜNG"

            # 4) backup: dùng anomaly_score với ngưỡng nhạy hơn
            elif anomaly_score >= 45:
                label = "ĐẮT BẤT THƯỜNG" if resid > 0 else "RẺ BẤT THƯỜNG"

            else:
                label = "BÌNH THƯỜNG"

            # -------------------------
            # Bước 7: Hiển thị kết quả (chỉ nhãn)
            # -------------------------
            st.markdown("---")
            if label == "ĐẮT BẤT THƯỜNG":
                st.error("🚨 **ĐẮT BẤT THƯỜNG** — mức giá cao hơn đáng kể so với mặt bằng phân khúc.")
            elif label == "RẺ BẤT THƯỜNG":
                st.error("🚨 **RẺ BẤT THƯỜNG** — mức giá thấp hơn đáng kể so với phân khúc.")
            else:
                st.success("✔ **BÌNH THƯỜNG** — mức giá phù hợp so với thị trường.")

elif choice == 'Danh sách xe giá bất thường': 

    st.write("### Danh sách các xe bất thường trong tập dữ liệu")

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
    # --- sửa: xử lý std == 0 và thay thế inf ---
    df_local["Residual_z"] = (df_local["Residual"] - df_local["mean"]) / df_local["std"]
    df_local["Residual_z"] = df_local["Residual_z"].replace([np.inf, -np.inf], np.nan)

    # --- 4. Tính các cờ vi phạm và điểm bất thường tổng ---
    # safe checks: only use columns if they exist after join
    # prepare default zero columns
    df_local["_minmax_violation"] = 0
    df_local["_p10p90_violation"] = 0

    # check existence of stat columns
    has_min = "min" in df_local.columns
    has_max = "max" in df_local.columns
    has_p10 = "p10" in df_local.columns
    has_p90 = "p90" in df_local.columns

    # compute minmax violation safely
    cond_min = pd.Series(False, index=df_local.index)
    cond_max = pd.Series(False, index=df_local.index)
    if has_min:
        cond_min = pd.notna(df_local["min"]) & (df_local["Giá"] < df_local["min"])
    if has_max:
        cond_max = pd.notna(df_local["max"]) & (df_local["Giá"] > df_local["max"])
    df_local.loc[cond_min | cond_max, "_minmax_violation"] = 1

    # compute percentile violation safely (P10-P90)
    cond_p10 = pd.Series(False, index=df_local.index)
    cond_p90 = pd.Series(False, index=df_local.index)
    if has_p10:
        cond_p10 = pd.notna(df_local["p10"]) & (df_local["Giá"] < df_local["p10"])
    if has_p90:
        cond_p90 = pd.notna(df_local["p90"]) & (df_local["Giá"] > df_local["p90"])
    df_local.loc[cond_p10 | cond_p90, "_p10p90_violation"] = 1

    # compute Residual_z safely (if mean/std exist)
    # already computed earlier; ensure no inf
    df_local["Residual_z"] = df_local["Residual_z"].replace([np.inf, -np.inf], np.nan)

    # compute residual score scaled to 0-100 using cap_z
    cap_z = 5.0
    df_local["_residual_score"] = df_local["Residual_z"].abs().fillna(0).clip(upper=cap_z) / cap_z * 100
    df_local["_minmax_score"] = df_local["_minmax_violation"] * 100
    df_local["_p10p90_score"] = df_local["_p10p90_violation"] * 100

    # anomaly score with weights w1=0.4, w2=0.4, w3=0.2
    w1, w2, w3 = 0.40, 0.40, 0.20
    df_local["_anomaly_score"] = (
        w1 * df_local["_residual_score"] +
        w2 * df_local["_minmax_score"] +
        w3 * df_local["_p10p90_score"]
    )
    df_local["_anomaly_score"] = df_local["_anomaly_score"].clip(0, 100)

    # --- 5. Lọc và hiển thị kết quả ---
    # điều kiện bất thường:
    #  - ưu tiên minmax hoặc percentile violations
    #  - hoặc residual_z vượt ±2
    #  - hoặc anomaly_score >= 60 (backup)
    cond_minmax = df_local["_minmax_violation"] == 1
    cond_percentile = df_local["_p10p90_violation"] == 1
    cond_residualz = df_local["Residual_z"].abs() >= 2
    cond_score = df_local["_anomaly_score"] >= 60

    df_abnormal = df_local[cond_minmax | cond_percentile | cond_residualz | cond_score].copy()

    # thêm cột phân loại hướng (dựa trên residual)
    def decide_label(row):
        # nếu vi phạm minmax -> định hướng theo seg min/max
        if row["_minmax_violation"] == 1:
            if pd.notna(row.get("min")) and row["Giá"] < row["min"]:
                return "RẺ BẤT THƯỜNG"
            if pd.notna(row.get("max")) and row["Giá"] > row["max"]:
                return "ĐẮT BẤT THƯỜNG"
        # nếu vi phạm percentile
        if row["_p10p90_violation"] == 1:
            if pd.notna(row.get("p10")) and row["Giá"] < row["p10"]:
                return "RẺ BẤT THƯỜNG"
            if pd.notna(row.get("p90")) and row["Giá"] > row["p90"]:
                return "ĐẮT BẤT THƯỜNG"
        # else use residual sign
        if pd.notna(row["Residual"]):
            return "ĐẮT BẤT THƯỜNG" if row["Residual"] > 0 else "RẺ BẤT THƯỜNG"
        return "BÌNH THƯỜNG"

    if not df_abnormal.empty:
        df_abnormal["Nhận định"] = df_abnormal.apply(decide_label, axis=1)

    if df_abnormal.empty:
        st.success("✔ Không có xe bất thường trong dataset.")
    else:
        st.error(f"💥 Có {len(df_abnormal)} xe bất thường:")
        display_cols = [
            "Thương hiệu","Dòng xe","Loại xe",
            "Giá","Giá dự đoán","Residual","Residual_z",
            "_anomaly_score","Nhận định"
        ]
        st.dataframe(
            df_abnormal[display_cols].sort_values("_anomaly_score", ascending=False)
        )
        csv_bytes = df_abnormal.to_csv(index=False).encode("utf-8")
        st.download_button("Tải toàn bộ danh sách bất thường (.csv)", csv_bytes, file_name="xe_bat_thuong.csv")