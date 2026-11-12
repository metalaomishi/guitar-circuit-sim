# app.py
import streamlit as st
import json
import os
import matplotlib.pyplot as plt

from src.core.guitar_engine import GuitarEngine

# ==========================================
# 1. データの初期化
# ==========================================

# JSONデータの読み込み
if not os.path.exists("data/pickups.json"):
    default_data = [
        {"name": "Seymour Duncan SH-1 '59", "L_pu": 3.9, "R_pu": 7600, "C_pu": 1.36e-10},
        {"name": "Seymour Duncan SH-4 JB", "L_pu": 8.1, "R_pu": 16400, "C_pu": 1.28e-10},
        {"name": "Fender Standard Single", "L_pu": 3.0, "R_pu": 6000, "C_pu": 1.00e-10}
    ]
    with open("data/pickups.json", "w", encoding="utf-8") as f_json:
        json.dump(default_data, f_json, indent=2)

with open("data/pickups.json", "r", encoding="utf-8") as f_json:
    pickup_data = json.load(f_json)
pu_names = [pu["name"] for pu in pickup_data]

# E12系列のコンデンサ基準値（μF）
e12_values = [0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.039, 0.047, 0.056, 0.068, 0.082, 0.100]

# ==========================================
# 2. コールバックの定義
# ==========================================

def on_model_change():
    """ギターの種類がドロップダウンで変更されたときに走る処理"""
    if st.session_state.model_selection == "Stratocaster":
        st.session_state.base_cap_selection = 0.047
    else:
        st.session_state.base_cap_selection = 0.022
    st.session_state.drift_value = 0.0  # ギター変更時は誤差もリセット

# コンデンサ容量復帰ボタンが押されたときの処理
def reset_to_normal_callback():
    """現在のギターの種類に応じて、コンデンサ値と誤差をノーマルに戻す"""
    if st.session_state.model_selection == "Stratocaster":
        st.session_state.base_cap_selection = 0.047
    else:
        st.session_state.base_cap_selection = 0.022
    st.session_state.drift_value = 0.0

# ==========================================
# 3. ページの設定
# ==========================================
st.set_page_config(page_title="Guitar Circuit Simulator", layout="wide")

# ==========================================
# 4. Webアプリ・セッション状態の管理（比較機能用）
# ==========================================
if "saved_curves" not in st.session_state:
    st.session_state.saved_curves = []

# ==========================================
# 5. セッション状態 (session_state) の初期化
# ==========================================

# コンデンサ基準値（セレクトボックス）の初期状態を管理
if "base_cap_selection" not in st.session_state:
    st.session_state.base_cap_selection = 0.022  # 初期値はLes Paulに合わせる

# 容量誤差（スライダー）の初期状態を管理
if "drift_value" not in st.session_state:
    st.session_state.drift_value = 0.0

# ==========================================
# 6. Web UI 画面レイアウト
# ==========================================
st.title("🎸 エレキギター回路・物理シミュレータ (v3.0 Web)")
st.caption("トーンコンデンサの容量ズレや内部損失(ESR)が、周波数特性に与える影響をリアルタイムに可視化します。")

# 画面を左（コントロール）と右（グラフ）に分割
col_ctrl, col_graph = st.columns([1, 2])

with col_ctrl:
    st.header("🎛 ️コントロールパネル")
    
    # 1. 基本構成
    st.subheader("基本構成", divider="gray")
    model_type = st.selectbox(
        "配線構造タイプ",
        ["Les Paul", "Stratocaster", "Flying V"],
        key="model_selection",
        on_change=on_model_change
    )

    # 2. ピックアップ選択 (JSON連動)
    selected_pu_name = st.selectbox("ピックアップ選択", pu_names)
    pu_param = next(pu for pu in pickup_data if pu["name"] == selected_pu_name)
    
    st.text(f"【PUスペック】 直流抵抗: {pu_param['R_pu']/1000:.1f}kΩ / インダクタンス: {pu_param['L_pu']:.1f}H")
    
    # 3. コントロールノブ操作
    st.subheader("コントロールノブ操作", divider=True)
    vol_pos = st.slider("Volume Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    tone_pos = st.slider("Tone Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)

    tab1, tab2, tab3 = st.tabs(["Modify", "Capacitor", "Memory"])

    # 4. モディファイ・特殊オプション
    with tab1:
        st.subheader("モディファイ・特殊オプション", divider=True)

        is_treble_bleed = st.checkbox("スムーステーパー (Treble Bleed) 有効化")
        has_esr = st.checkbox("疑似オイルコンデンサ (ESR内部損失 120Ωを付加)")
        is_b_curve = st.checkbox("Bカーブポット (リニア特性) に変更")

        pot_curve = "B-Curve (Linear)" if is_b_curve else "A-Curve (Audio/Log)"

        # --- オイルコンデンサのESR（直列抵抗）成分の処理 ---
        # 聴感変化を体感しやすいよう、オイルの不完全な損失成分(あえて実効的な120Ω)をシミュレート
        R_esr = 120.0 if has_esr else 0.0

    # 5. コンデンサ
    with tab2:
        st.subheader("コンデンサ設定", divider=True)

        # コンデンサ容量復帰ボタン
        st.button("コンデンサ容量を戻す", on_click=reset_to_normal_callback, use_container_width=True)

        # コンデンサ基準値のセレクトボックス (keyとindexを連動)
        # 現在のセッション状態の数値が、e12_valuesリストの何番目にあるかを自動計算して初期選択にする
        current_index = e12_values.index(st.session_state.base_cap_selection)
        base_cap_uf = st.selectbox(
            "E12系列 コンデンサ基準値 (μF)", 
            e12_values, 
            index=current_index,
            key="base_cap_selection"
        )

        # 容量誤差スライダー (keyを連動)
        cap_drift_pct = st.slider(
            "容量の個体誤差・経年変化 (±20%)", 
            min_value=-20.0, 
            max_value=20.0, 
            step=0.5,
            key="drift_value"
        )
        
        # E12ベース容量に±20%の誤差を適用
        actual_cap_uf = base_cap_uf * (1.0 + (cap_drift_pct / 100.0))
        st.info(f"実効容量: {actual_cap_uf:.4f} μF")

    # 6. 重ね合わせ比較コントロール
    with tab3:    
        st.subheader("💾 波形比較メモリ", divider=True)

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("現在の波形を保存"):
                result = GuitarEngine.calculate_response(model_type, vol_pos, tone_pos, actual_cap_uf, pot_curve, pu_param, is_treble_bleed, R_esr)
                label = f"{model_type} V:{vol_pos:.0f} T:{tone_pos:.0f} {actual_cap_uf:.3f}uF {'+ESR' if has_esr else ''}"
                st.session_state.saved_curves.append({"frequency": result["frequency"], "gain": result["gain"], "label": label})
                st.rerun()
        with c_btn2:
            if st.button("メモリをクリア"):
                st.session_state.saved_curves.clear()
                st.rerun()


with col_graph:
    st.header("📊 周波数特性グラフ")
    # actual_cap_uf = base_cap_uf * (1.0 + (cap_drift_pct / 100.0))
    
    # ライブデータの計算
    result = GuitarEngine.calculate_response(model_type, vol_pos, tone_pos, actual_cap_uf, pot_curve, pu_param, is_treble_bleed, R_esr)
    
    # Matplotlibによる描画作成
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xscale('log')
    ax.set_xlim(40, 15000)
    ax.set_ylim(-40, 5)
    ax.grid(True, which="both", ls="--", color='0.8')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    
    # 保存された過去の線をプロット
    for saved in st.session_state.saved_curves:
        ax.plot(saved["frequency"], saved["gain"], label=saved["label"], alpha=0.5, linestyle="--")
        
    # 現在のライブデータを赤の太線でプロット
    ax.plot(result["frequency"], result["gain"], color='red', linewidth=3, label="LIVE (Current settings)")
    ax.legend(loc='lower left', fontsize=9)
    
    # Web画面にグラフを出力
    st.pyplot(fig)
