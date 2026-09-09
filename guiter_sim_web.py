import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ページの設定（ブラウザのタブに表示されるタイトルなど）
st.set_page_config(page_title="Professional Guitar Circuit Simulator", layout="wide")

# 周波数軸の設定 (20Hz 〜 20kHz)
f = np.logspace(1, 4.3, 300) 
w = 2 * np.pi * f

# ==========================================
# 1. 物理シミュレーション・エンジン（v3.0のロジックを完全流用）
# ==========================================
def calculate_response(model_type, vol_pos, tone_pos, base_cap_uf, cap_drift_pct, pot_curve, pu_param, is_treble_bleed, has_esr):
    L_pu = pu_param["L_pu"]
    R_pu = pu_param["R_pu"]
    C_pu = pu_param["C_pu"]

    R_vol_max, R_tone_max = (250e3, 250e3) if model_type == "Stratocaster" else (500e3, 500e3)
    C_total_load = C_pu + 500e-12  # シールドケーブル500pF
    actual_cap_uf = base_cap_uf * (1.0 + (cap_drift_pct / 100.0))
    C_tone = actual_cap_uf * 1e-6
    R_amp = 1e6

    v_norm, t_norm = vol_pos / 10.0, tone_pos / 10.0
    if pot_curve == "A-Curve (Audio/Log)":
        v_ratio = v_norm ** 3
        t_ratio = t_norm ** 3 if tone_pos > 0 else 0.0001
    else:
        v_ratio = v_norm
        t_ratio = t_norm if tone_pos > 0 else 0.0001
    
    R_esr = 120.0 if has_esr else 0.0
    R_t = R_tone_max * t_ratio if tone_pos > 0 else 0.1
    Z_tone_branch = R_t + R_esr + (1 / (1j * w * C_tone))

    R1, R2 = R_vol_max * (1.0 - v_ratio), R_vol_max * v_ratio

    if is_treble_bleed and R1 > 0:
        C_bleed, R_bleed = 1000e-12, 220e3
        Z_bleed_branch = 1 / (1 / R_bleed + 1j * w * C_bleed)
        Z_series = 1 / (1 / R1 + 1 / Z_bleed_branch)
    else:
        Z_series = R1

    if model_type == "Stratocaster": 
        Z_load2 = 1 / (1 / (R2 + 1e-1) + 1 / R_amp) if R2 > 0 else 0
        Z_load1 = 1 / (1 / Z_tone_branch + 1 / (Z_series + Z_load2 + 1e-1) + 1j * w * C_total_load)
    else: 
        Z_load3 = 1 / (1 / Z_tone_branch + 1 / R_amp)
        Z_load2 = 1 / (1 / (R2 + 1e-1) + 1 / Z_load3) if R2 > 0 else 0
        Z_load1 = 1 / (1 / (Z_series + Z_load2 + 1e-1) + 1j * w * C_total_load)

    Z_pu = R_pu + 1j * w * L_pu
    V_output = (Z_load1 / (Z_pu + Z_load1)) * (Z_load2 / (Z_series + Z_load2 + 1e-1))
    return 20 * np.log10(np.abs(V_output))

# ==========================================
# 2. JSONデータの読み込み
# ==========================================
if not os.path.exists("pickups.json"):
    default_data = [
        {"name": "Seymour Duncan SH-1 '59", "L_pu": 3.9, "R_pu": 7600, "C_pu": 1.36e-10},
        {"name": "Seymour Duncan SH-4 JB", "L_pu": 8.1, "R_pu": 16400, "C_pu": 1.28e-10},
        {"name": "Fender Standard Single", "L_pu": 3.0, "R_pu": 6000, "C_pu": 1.00e-10}
    ]
    with open("pickups.json", "w", encoding="utf-8") as f_json:
        json.dump(default_data, f_json, indent=2)

with open("pickups.json", "r", encoding="utf-8") as f_json:
    pickup_data = json.load(f_json)
pu_names = [pu["name"] for pu in pickup_data]

# ==========================================
# 3. Webアプリ・セッション状態の管理（比較機能用）
# ==========================================
if "saved_curves" not in st.session_state:
    st.session_state.saved_curves = []

# ==========================================
# 4. Web UI 画面レイアウト
# ==========================================
st.title("🎸 エレキギター回路・物理シミュレータ (v3.0 Web)")
st.caption("トーンコンデンサの容量ズレや内部損失(ESR)が、周波数特性に与える影響をリアルタイムに可視化します。")

# 画面を左（コントロール）と右（グラフ）に分割
col_ctrl, col_graph = st.columns([1, 2])

with col_ctrl:
    st.header("🎛 ️コントロールパネル")
    
    # 1. 基本構成
    model_type = st.selectbox("配線構造タイプ", ["Les Paul", "Stratocaster", "Flying V"])
    pot_curve = st.selectbox("ポットのカーブ特性", ["A-Curve (Audio/Log)", "B-Curve (Linear)"])
    selected_pu_name = st.selectbox("ピックアップ選択", pu_names)
    pu_param = next(pu for pu in pickup_data if pu["name"] == selected_pu_name)
    
    st.text(f"【PUスペック】 直流抵抗: {pu_param['R_pu']/1000:.1f}kΩ / インダクタンス: {pu_param['L_pu']:.1f}H")
    # st.separator()
    
    # 2. オプション
    is_treble_bleed = st.checkbox("スムーステーパー (Treble Bleed) 有効化")
    has_esr = st.checkbox("疑似オイルコンデンサ (ESR内部損失 120Ωを付加)")
    # st.separator()
    
    # 3. ノブ操作
    vol_pos = st.slider("Volume Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    tone_pos = st.slider("Tone Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    # st.separator()
    
    # 4. コンデンサ
    e12_values = [0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.039, 0.047, 0.056, 0.068, 0.082, 0.100]
    base_cap_uf = st.selectbox("E12系列 コンデンサ基準値 (μF)", e12_values, index=4) # デフォルト0.022
    cap_drift_pct = st.slider("容量の個体誤差・経年変化 (±20%)", min_value=-20.0, max_value=20.0, value=0.0, step=0.5)
    
    actual_cap = base_cap_uf * (1.0 + (cap_drift_pct / 100.0))
    st.info(f"実効容量: {actual_cap:.4f} μF")
    # st.separator()
    
    # 5. 重ね合わせ比較コントロール
    st.subheader("💾 波形比較メモリ")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("現在の波形を保存"):
            current_gain = calculate_response(model_type, vol_pos, tone_pos, base_cap_uf, cap_drift_pct, pot_curve, pu_param, is_treble_bleed, has_esr)
            label = f"{model_type} V:{vol_pos:.0f} T:{tone_pos:.0f} {actual_cap:.3f}uF {'+ESR' if has_esr else ''}"
            st.session_state.saved_curves.append({"gain": current_gain, "label": label})
            st.rerun()
    with c_btn2:
        if st.button("メモリをクリア"):
            st.session_state.saved_curves.clear()
            st.rerun()

with col_graph:
    st.header("📊 周波数特性グラフ")
    
    # ライブデータの計算
    live_gain = calculate_response(model_type, vol_pos, tone_pos, base_cap_uf, cap_drift_pct, pot_curve, pu_param, is_treble_bleed, has_esr)
    
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
        ax.plot(f, saved["gain"], label=saved["label"], alpha=0.5, linestyle="--")
        
    # 現在のライブデータを赤の太線でプロット
    ax.plot(f, live_gain, color='red', linewidth=3, label="LIVE (Current settings)")
    ax.legend(loc='lower left', fontsize=9)
    
    # Web画面にグラフを出力
    st.pyplot(fig)
