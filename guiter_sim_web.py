import streamlit as st
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ページの設定（ブラウザのタブに表示されるタイトルなど）
st.set_page_config(page_title="Guitar Circuit Simulator", layout="wide")

# 周波数軸の設定 (20Hz 〜 20kHz)
f = np.logspace(1, 4.3, 300) 
w = 2 * np.pi * f
s = 1j * w # ラプラス変換の複素周波数領域での表現

# ==========================================
# 1. 物理シミュレーション・エンジン（v3.0）
# ==========================================
def calculate_response(model_type, vol_pos, tone_pos, cap_val_uf, pot_curve, pu_param, is_treble_bleed, cap_esr=0):
    # JSONから読み込んだピックアップ特性
    L_pu = pu_param["L_pu"]
    R_pu = pu_param["R_pu"]
    C_pu = pu_param["C_pu"]

    # ギターモデル（配線トポロジー）ごとのポット基本値
    if model_type == "Stratocaster":
        R_vol_max, R_tone_max = 250e3, 250e3
    else: # Les Paul / Flying V
        R_vol_max, R_tone_max = 500e3, 500e3

    R_amp = 1e6  # アンプ入力抵抗
    C_cable = 500e-12  # シールドケーブル容量500pF
    
    # --- トーンポットカーブ（Aカーブ vs Bカーブ）の処理 ---
    v_norm = vol_pos / 10.0
    t_norm = tone_pos / 10.0
    
    if pot_curve == "B-Curve (Linear)":
        v_ratio = v_norm if vol_pos > 0 else 0.0001
        t_ratio = t_norm if tone_pos > 0 else 0.0001
    else: # A-Curve (Audio/Log)
        v_ratio = v_norm ** 3 if vol_pos > 0 else 0.0001
        t_ratio = t_norm ** 3 if tone_pos > 0 else 0.0001
    
    C_tone = cap_val_uf * 1e-6
    R_tone = R_tone_max * t_ratio if tone_pos > 0 else 0.1
    # トーン回路のブランチインピーダンスにR_esrを直列合算
    Z_tone_branch = R_tone + cap_esr + (1 / (s * C_tone))

    RV1 = R_vol_max * (1.0 - v_ratio)
    RV2 = R_vol_max * v_ratio

    N = len(f) # 周波数点数

    # --- ハイパスフィルター（スムーステーパー）の処理 ---
    if is_treble_bleed:
        C_bleed = 1000e-12 # 定番の1000pF
        R_bleed = 220e3    # 定番の220kΩ
        Z_bleed_branch = 1 / ((1 / R_bleed) + (s * C_bleed))
        # RV1とハイパスが並列合流
        Z_series = 1 / ((1 / RV1) + (1 / Z_bleed_branch)) if RV1 > 0 else Z_bleed_branch
    else:
        Z_series = np.full(N, RV1)

    # サーキットを3段のラダー回路網と見立て6つの分岐インピーダンスを計算
    # 直列 1段目 (ピックアップ等価回路)
    Z1 = R_pu + (s * L_pu)  # ピックアップの等価インピーダンス（コイル抵抗 + インダクタンス）
    # 並列 1段目 (ピックアップ寄生容量)
    Y2 = s * C_pu  # ピックアップ寄生容量のアドミッタンス
    # トーン＆ボリュームコントロール回路
    if model_type == "Les Paul": # ギブソン系：トーンが先、ボリュームが後
        # 直列 2段目 (該当なし)
        Z3 = np.full(N, 0.0)
        # 並列 2段目 (トーン回路)
        Y4 = 1 / Z_tone_branch
        # 直列 3段目 (トーン回路の後にボリュームの分圧回路が入る)
        Z5 = Z_series
        # 並列 3段目 (ボリュームの分圧)
        Y6 = np.full(N, 1 / RV2)
    else: # ストラトキャスター・フライングV：ボリュームが先、トーンが後
        # 直列 2段目 (ボリュームの分圧回路)
        Z3 = Z_series
        # 並列 2段目 (ボリュームの分圧回路)
        Y4 = np.full(N, 1 / RV2)
        # 直列 3段目 (該当なし)
        Z5 = np.full(N, 0.0)
        # 並列 3段目 (トーン回路)
        Y6 = 1 / Z_tone_branch
    # シールドケーブルとアンプ入力の等価インピーダンスを加味
    Y6 = Y6 + (s * C_cable) + (1 / R_amp)  # シールドケーブルとアンプ入力の並列合成

    # NumPyによるF行列（伝達マトリクス）の構築と掛け算

    # 各素子の2x2行列の初期化 (全周波数分を一括作成)
    F1 = np.array([[np.ones(N), Z1],         [np.zeros(N), np.ones(N)]])
    F2 = np.array([[np.ones(N), np.zeros(N)], [Y2,         np.ones(N)]])
    F3 = np.array([[np.ones(N), Z3],         [np.zeros(N), np.ones(N)]])
    F4 = np.array([[np.ones(N), np.zeros(N)], [Y4,         np.ones(N)]])
    F5 = np.array([[np.ones(N), Z5],         [np.zeros(N), np.ones(N)]])
    F6 = np.array([[np.ones(N), np.zeros(N)], [Y6,         np.ones(N)]])

    # NumPyの軸を調整 (2, 2, N) -> (N, 2, 2) に変換して行列積を計算しやすくする
    F1 = np.moveaxis(F1, -1, 0)
    F2 = np.moveaxis(F2, -1, 0)
    F3 = np.moveaxis(F3, -1, 0)
    F4 = np.moveaxis(F4, -1, 0)
    F5 = np.moveaxis(F5, -1, 0)
    F6 = np.moveaxis(F6, -1, 0)

    # 左から順番に行列を掛け算 ( @ はNumPyの行列積演算子 )
    F_total = F1 @ F2 @ F3 @ F4 @ F5 @ F6

    # 回路合成した時の伝達関数を算出
    A_m = F_total[:, 0, 0]
    H_m = 1 / A_m  # 伝達関数 (出力開放時)

    # ピックアップ出力電圧を1Vとする
    V_pu = 1.0

    # 最終的な出力電圧（＝ピックアップ出力電圧を1Vとした場合のアンプ入力電圧）
    V_output = V_pu * H_m  # 最終的な出力電圧（＝アンプ入力電圧）
    # V_output = V3  
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
# 4. セッション状態 (session_state) の初期化
# ==========================================
# E12系列のコンデンサ基準値（μF）
e12_values = [0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.039, 0.047, 0.056, 0.068, 0.082, 0.100]

# コンデンサ基準値（セレクトボックス）の初期状態を管理
if "base_cap_selection" not in st.session_state:
    st.session_state.base_cap_selection = 0.022  # 初期値はLes Paulに合わせる

# 容量誤差（スライダー）の初期状態を管理
if "drift_value" not in st.session_state:
    st.session_state.drift_value = 0.0

def on_model_change():
    """ギターの種類がドロップダウンで変更されたときに走る処理"""
    if st.session_state.model_selection == "Stratocaster":
        st.session_state.base_cap_selection = 0.047
    else:
        st.session_state.base_cap_selection = 0.022
    st.session_state.drift_value = 0.0  # ギター変更時は誤差もリセット

# ==========================================
# 5. Web UI 画面レイアウト
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
    
    # 3. モディファイ・特殊オプション
    st.subheader("モディファイ・特殊オプション", divider=True)

    is_treble_bleed = st.checkbox("スムーステーパー (Treble Bleed) 有効化")
    has_esr = st.checkbox("疑似オイルコンデンサ (ESR内部損失 120Ωを付加)")
    pot_curve = st.selectbox("ポットのカーブ特性", ["A-Curve (Audio/Log)", "B-Curve (Linear)"])

    # --- オイルコンデンサのESR（直列抵抗）成分の処理 ---
    # 聴感変化を体感しやすいよう、オイルの不完全な損失成分(あえて実効的な120Ω)をシミュレート
    R_esr = 120.0 if has_esr else 0.0
    
    # 4. コントロールノブ操作
    st.subheader("コントロールノブ操作", divider=True)
    vol_pos = st.slider("Volume Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    tone_pos = st.slider("Tone Control (0 - 10)", min_value=0.0, max_value=10.0, value=10.0, step=0.5)
    
    # 5. コンデンサ
    st.subheader("コンデンサ設定", divider=True)

    # コンデンサ容量復帰ボタンが押されたときの処理
    def reset_to_normal_callback():
        """現在のギターの種類に応じて、コンデンサ値と誤差をノーマルに戻す"""
        if st.session_state.model_selection == "Stratocaster":
            st.session_state.base_cap_selection = 0.047
        else:
            st.session_state.base_cap_selection = 0.022
        st.session_state.drift_value = 0.0

    # コンデンサ容量復帰ボタンの配置
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
    st.subheader("💾 波形比較メモリ", divider=True)

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("現在の波形を保存"):
            current_gain = calculate_response(model_type, vol_pos, tone_pos, actual_cap_uf, pot_curve, pu_param, is_treble_bleed, R_esr)
            label = f"{model_type} V:{vol_pos:.0f} T:{tone_pos:.0f} {actual_cap_uf:.3f}uF {'+ESR' if has_esr else ''}"
            st.session_state.saved_curves.append({"gain": current_gain, "label": label})
            st.rerun()
    with c_btn2:
        if st.button("メモリをクリア"):
            st.session_state.saved_curves.clear()
            st.rerun()

with col_graph:
    st.header("📊 周波数特性グラフ")
    # actual_cap_uf = base_cap_uf * (1.0 + (cap_drift_pct / 100.0))
    
    # ライブデータの計算
    live_gain = calculate_response(model_type, vol_pos, tone_pos, actual_cap_uf, pot_curve, pu_param, is_treble_bleed, R_esr)
    
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
