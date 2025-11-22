# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import os
from difflib import SequenceMatcher
from math import ceil

# ---------------- Config ----------------
DATA_PATH = "data_motobikes.xlsx"
COSINE_PKL = "cosine_sim_model.pkl"
UPLOADED_IMAGE = "b12bca47-fea2-499d-80f1-1915896b8525.png"

st.set_page_config(page_title="Gợi ý xe bằng Cosine", layout="wide")

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