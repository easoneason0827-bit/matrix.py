import streamlit as st
import numpy as np
import scipy.linalg as la
import pandas as pd

# 設定網頁標題與組員資訊 [cite: 1, 2, 3, 4]
st.set_page_config(page_title="矩陣運算網站", page_icon="🧮")
st.title("🧮 線性代數矩陣運算網站")
st.markdown("### 組員：11428240 高翊豪 | 11428205 游郁晨")
st.write("動機：加速線性代數運算，提供類似工程計算機的功能 [cite: 5, 6]")

# --- 工具函數：解析矩陣輸入 ---
def parse_matrix(input_str, rows, cols):
    try:
        # 將輸入的字串轉換為數值列表
        data = [float(x) for x in input_str.split()]
        if len(data) != rows * cols:
            return None, f"錯誤：輸入數據數量 ({len(data)}) 與設定的大小 ({rows}x{cols}={rows*cols}) 不符。"
        return np.array(data).reshape(rows, cols), None
    except ValueError:
        return None, "錯誤：請確認輸入的都是數字。"

# --- 側邊欄：矩陣維度設定  ---
st.sidebar.header("矩陣維度設定")
m = st.sidebar.number_input("矩陣 A 列數 (m)", min_value=1, value=3)
n = st.sidebar.number_input("矩陣 A 行數 (n)", min_value=1, value=3)
st.sidebar.markdown("---")
p = st.sidebar.number_input("矩陣 B 列數 (p)", min_value=1, value=3)
q = st.sidebar.number_input("矩陣 B 行數 (q)", min_value=1, value=3)
st.sidebar.markdown("---")
st.sidebar.caption("提示：可在表格內直接編輯或貼上資料 (copy/paste)")

# --- 主畫面：輸入矩陣 A ---
tabs = st.tabs(["單矩陣 A", "雙矩陣 A/B", "進階運算與說明"])

with tabs[0]:
    st.subheader(f"1. 輸入矩陣 A ({m}x{n})")
    # 使用 DataFrame 與 data_editor 提供表格式編輯
    default_a = np.arange(1, m * n + 1).reshape(m, n)
    df_a = pd.DataFrame(default_a)
    edited_a = st.data_editor(df_a, num_rows="fixed", width='stretch', key='matrix_a')
    try:
        matrix_a = edited_a.values.astype(float)
        st.write("矩陣 A：")
        st.dataframe(edited_a)
    except Exception:
        st.error("請確認矩陣 A 的資料為數值")

    st.info("單一矩陣運算 (針對 A)")
    with st.form("A_ops"):
        c1, c2, c3 = st.columns(3)
        with c1:
            det_btn = st.form_submit_button("計算 det(A)")
        with c2:
            trans_btn = st.form_submit_button("A 的轉置矩陣")
        with c3:
            inv_btn = st.form_submit_button("A inverse")

        c4, c5 = st.columns(2)
        with c4:
            lu_btn = st.form_submit_button("LU 分解")
        with c5:
            st.markdown("**解 Ax = b**")
            input_b_vec = st.text_input(f"輸入向量 b ({m} 個數字)", "1 1 1")
            solve_btn = st.form_submit_button("求解 x")

    # 處理表單提交
    if det_btn:
        if m == n:
            with st.spinner("計算中..."):
                det_val = np.linalg.det(matrix_a)
            st.success(f"det(A) = {det_val:.4f}")
        else:
            st.error("只有方陣 (m=n) 才能計算行列式")

    if trans_btn:
        with st.spinner("轉置中..."):
            st.write(matrix_a.T)

    if inv_btn:
        if m == n:
            try:
                with st.spinner("計算反矩陣..."):
                    inv_a = np.linalg.inv(matrix_a)
                st.write(inv_a)
            except np.linalg.LinAlgError:
                st.error("此矩陣為奇異矩陣 (Singular)，無反矩陣")
        else:
            st.error("只有方陣才能計算反矩陣")

    if lu_btn:
        with st.spinner("LU 分解中..."):
            P, L, U = la.lu(matrix_a)
        st.write("P (Permutation):", P)
        st.write("L (Lower):", L)
        st.write("U (Upper):", U)

    if solve_btn:
        try:
            b_vec = np.array([float(x) for x in input_b_vec.split()])
            if len(b_vec) == m:
                if m == n:
                    with st.spinner("求解中..."):
                        x = np.linalg.solve(matrix_a, b_vec)
                    st.success(f"x = {x}")
                else:
                    with st.spinner("計算最小二乘解..."):
                        x, residuals, rank, s = np.linalg.lstsq(matrix_a, b_vec, rcond=None)
                    st.warning("A 不是方陣，顯示最小二乘解：")
                    st.write(x)
            else:
                st.error(f"向量 b 的長度必須為 {m}")
        except Exception as e:
            st.error(f"無法求解: {e}")

with tabs[1]:
    st.subheader(f"3. 輸入矩陣 B ({p}x{q})")
    default_b = np.arange(1, p * q + 1).reshape(p, q)
    df_b = pd.DataFrame(default_b)
    edited_b = st.data_editor(df_b, num_rows="fixed", width='stretch', key='matrix_b')
    try:
        matrix_b = edited_b.values.astype(float)
        st.write("矩陣 B：")
        st.dataframe(edited_b)
    except Exception:
        st.error("請確認矩陣 B 的資料為數值")

    st.info("雙矩陣運算 (A 與 B)")
    with st.form("AB_ops"):
        op_col1, op_col2, op_col3, op_col4 = st.columns(4)
        with op_col1:
            add_btn = st.form_submit_button("計算 A + B")
        with op_col2:
            sub_btn = st.form_submit_button("計算 A - B")
        with op_col3:
            mul_ab_btn = st.form_submit_button("計算 AB (矩陣乘法)")
        with op_col4:
            mul_ba_btn = st.form_submit_button("計算 BA (矩陣乘法)")

    # 處理 AB 操作
    if add_btn:
        if (m, n) == (p, q):
            st.write(matrix_a + matrix_b)
        else:
            st.error(f"維度不符：A({m}x{n}) != B({p}x{q})")

    if sub_btn:
        if (m, n) == (p, q):
            st.write(matrix_a - matrix_b)
        else:
            st.error(f"維度不符：A({m}x{n}) != B({p}x{q})")

    if mul_ab_btn:
        if n == p:
            st.write(np.dot(matrix_a, matrix_b))
        else:
            st.error(f"無法相乘：A的列數({n}) != B的列數({p})")

    if mul_ba_btn:
        if q == m:
            st.write(np.dot(matrix_b, matrix_a))
        else:
            st.error(f"無法相乘：B的列數({q}) != A的列數({m})")

with tabs[2]:
    st.subheader("進階運算與說明")
    st.markdown("- 使用上：在表格內直接編輯矩陣，完成後按相對應的按鈕提交。")
    st.markdown("- 若要快速填入測試資料，請調整左側的矩陣尺寸，表格會顯示預設數值。")
    with st.expander("進階運算 (SVD / 特徵值 / 條件數 / Rank)"):
        if st.button("計算進階項目 (A)"):
            try:
                with st.spinner("計算中..."):
                    u, s, vh = np.linalg.svd(matrix_a, full_matrices=False)
                    eigvals = None
                    if m == n:
                        eigvals = np.linalg.eigvals(matrix_a)
                    cond = np.linalg.cond(matrix_a)
                    rank = np.linalg.matrix_rank(matrix_a)
                st.write("奇異值：", s)
                if eigvals is not None:
                    st.write("特徵值：", eigvals)
                st.write("條件數：", cond)
                st.write("Rank：", rank)
            except Exception as e:
                st.error(f"計算失敗: {e}")