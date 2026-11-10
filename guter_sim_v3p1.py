import tkinter as tk
from tkinter import ttk
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 周波数軸の設定 (20Hz 〜 20kHz)
f = np.logspace(1, 4.3, 300) 
w = 2 * np.pi * f
s = 1j * w # ラプラス変換の複素周波数領域での表現

# ==========================================
# 1. 物理シミュレーション・エンジン（v3.0）
# ==========================================
class GuitarEngineV3:
    @staticmethod
    def calculate(model_type, vol_pos, tone_pos, cap_val_uf, pu_param, is_b_curve, is_treble_bleed, cap_esr=0):
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
        
        if is_b_curve: # B-Curve (Linear)
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
# 2. GUI アプリケーション本体
# ==========================================
class GuitarSimAppV3:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Guitar Circuit Simulator (v3.1)")
        self.root.geometry("1100x700")

        # JSONファイルの読み込み
        self.load_pickups_json()
        self.saved_curves = []  # ホールド比較用の過去データ保持リスト

        # 左右のメイン分割
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 左側：コントロールパネル枠 ---
        control_frame = ttk.Frame(main_frame, padding=10, relief="solid", borderwidth=1)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 1. 基本構成設定
        ttk.Label(control_frame, text="1. 基本配線 & カーブ設定", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W, pady=(5,5))
        self.model_var = tk.StringVar(value="Les Paul")
        self.model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=["Stratocaster", "Les Paul", "Flying V"], state="readonly")
        self.model_combo.pack(fill=tk.X, pady=(0, 5))
        self.model_combo.bind("<<ComboboxSelected>>", self.update_plot)

        # 2. ピックアップ選択 (JSON連動)
        ttk.Label(control_frame, text="2. ピックアップ選択:", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W, pady=(0,5))
        self.pu_names = [pu["name"] for pu in self.pickup_data]
        self.pu_var = tk.StringVar(value=self.pu_names[0])
        self.pu_combo = ttk.Combobox(control_frame, textvariable=self.pu_var, values=self.pu_names, state="readonly")
        self.pu_combo.pack(fill=tk.X, pady=(0, 15))
        self.pu_combo.bind("<<ComboboxSelected>>", self.update_plot)

        # 3. モディファイ・特殊オプション
        ttk.Label(control_frame, text="3. モディファイ・特殊オプション:", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W, pady=(0,5))
        self.hp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="スムーステーパー改造 (Treble Bleed)", variable=self.hp_var, command=self.update_plot).pack(anchor=tk.W)
        self.esr_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="オイルコンデンサ交換 (Add ESR)", variable=self.esr_var, command=self.update_plot).pack(anchor=tk.W)
        self.curve_var = tk.BooleanVar(value=False) # "A-Curve (Audio/Log)"
        ttk.Checkbutton(control_frame, text="ポットBカーブ改造 (Linear)", variable=self.curve_var, command=self.update_plot).pack(anchor=tk.W, pady=(0, 15))

        # 4. コントロールノブ
        ttk.Label(control_frame, text="Volume Control (0 - 10):", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W)
        self.vol_scale = ttk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=self.update_plot)
        self.vol_scale.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(control_frame, text="Tone Control (0 - 10):", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W)
        self.tone_scale = ttk.Scale(control_frame, from_=0, to=10, orient=tk.HORIZONTAL, command=self.update_plot)
        self.tone_scale.pack(fill=tk.X, pady=(0, 15))

        # 5. E12系列コンデンサ & 偏差スライダー
        ttk.Label(control_frame, text="トーンコンデンサ容量 (μF):", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W, pady=(0,5))
        self.e12_values = [0.010, 0.012, 0.015, 0.018, 0.022, 0.027, 0.033, 0.039, 0.047, 0.056, 0.068, 0.082, 0.100]
        self.e12_var = tk.DoubleVar(value=0.022)
        self.e12_combo = ttk.Combobox(control_frame, textvariable=self.e12_var, values=self.e12_values, state="readonly")
        self.e12_combo.pack(fill=tk.X, pady=(0, 5))
        self.e12_combo.bind("<<ComboboxSelected>>", self.update_plot)

        ttk.Label(control_frame, text="容量の個体誤差/経年変化 (±20%):", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W)
        self.cap_scale = ttk.Scale(control_frame, from_=-20, to=20, orient=tk.HORIZONTAL, command=self.update_plot)
        self.cap_scale.pack(fill=tk.X, pady=(0, 5))
        
        # コンデンサ容量復帰ボタン
        self.reset_cap_btn = ttk.Button(control_frame, text="容量をノーマルに戻す", command=self.reset_cap)
        self.reset_cap_btn.pack(fill=tk.X, pady=(0, 15))

        self.cap_label = ttk.Label(control_frame, text="現在の値: 0.022 μF")
        self.cap_label.pack(anchor=tk.W, pady=(0, 20))

        # 6. ホールド比較（重ね合わせ）パネル
        ttk.Label(control_frame, text="6. 波形比較コントロール", font=("MS Gothic", 10, "bold")).pack(anchor=tk.W, pady=(5,5))
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="現在の波形を保存", command=self.save_current_curve).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_frame, text="クリア", command=self.clear_saved_curves).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # 初期スライダー値セット
        self.vol_scale.set(10.0)
        self.tone_scale.set(10.0)
        self.cap_scale.set(0.0)

        # スペック表示枠
        self.param_label = ttk.Label(control_frame, text="", justify=tk.LEFT, foreground="gray")
        self.param_label.pack(anchor=tk.W, pady=(15, 0))

        # --- 右側：グラフ枠 ---
        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.init_plot()

    def load_pickups_json(self):
        filename = "./pickups.json"
        if not os.path.exists(filename):
            # ファイルがない場合のフォールバック（自動生成）
            default_data = [
                {"name": "Seymour Duncan SH-1 '59", "L_pu": 3.9, "R_pu": 7600, "C_pu": 1.36e-10},
                {"name": "Seymour Duncan SH-4 JB", "L_pu": 8.1, "R_pu": 16400, "C_pu": 1.28e-10},
                {"name": "Fender Standard Single", "L_pu": 3.0, "R_pu": 6000, "C_pu": 1.00e-10}
            ]
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)
        with open(filename, "r", encoding="utf-8") as f:
            self.pickup_data = json.load(f)

    def init_plot(self):
        self.ax.set_xscale('log')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Gain (dB)')
        self.ax.set_xlim(40, 15000)
        self.ax.set_ylim(-40, 5)
        self.ax.grid(True, which="both", ls="--", color='0.8')
        self.update_plot()

    def reset_cap(self):
        model = self.model_var.get()
        if model == "Stratocaster":
            self.e12_var.set(0.047)
        else:
            self.e12_var.set(0.022)
        self.cap_scale.set(0.0)
        self.update_plot()

    def save_current_curve(self):
        # 現在のパラメータと計算結果をスナップショットとして保存
        model = self.model_var.get()
        vol = self.vol_scale.get()
        tone = self.tone_scale.get()
        base_cap = self.e12_var.get()
        cap_drift_pct = self.cap_scale.get()
        is_b_curve = self.curve_var.get()
        pu_name = self.pu_var.get().split()[0]
        esr_status = "+ESR" if self.esr_var.get() else ""
        
        curve_type = "B-Curve (Linear)" if is_b_curve else "A-Curve (Audio/Log)"
        # E12ベース容量に±20%の誤差を適用
        actual_cap_uf = base_cap * (1.0 + (cap_drift_pct / 100.0))
        
        label_text = f"{model}({pu_name}) V:{vol:.0f} T:{tone:.0f} {actual_cap_uf:.3f}uF {curve_type} {esr_status}"

        # 描画用の現在のゲインデータを取得
        selected_pu_name = self.pu_var.get()
        pu_param = next(pu for pu in self.pickup_data if pu["name"] == selected_pu_name)
        gain = GuitarEngineV3.calculate(model, vol, tone, actual_cap_uf, pu_param, self.curve_var.get(), self.hp_var.get(), self.esr_var.get())
        self.saved_curves.append({"gain": gain, "label": label_text})
        self.update_plot()

    def clear_saved_curves(self):
        self.saved_curves.clear()
        self.update_plot()

    def update_plot(self, *args):
        model = self.model_var.get()
        vol = self.vol_scale.get()
        tone = self.tone_scale.get()
        base_cap = self.e12_var.get()
        cap_drift_pct = self.cap_scale.get()
        is_hypass = self.hp_var.get()
        has_esr = self.esr_var.get()
        is_b_curve = self.curve_var.get()
        selected_pu_name = self.pu_var.get()
        pu_param = next(pu for pu in self.pickup_data if pu["name"] == selected_pu_name)

        curve_type = "B-Curve (Linear)" if is_b_curve else "A-Curve (Audio/Log)"
        # E12ベース容量に±20%の誤差を適用
        actual_cap_uf = base_cap * (1.0 + (cap_drift_pct / 100.0))

        # --- オイルコンデンサのESR（直列抵抗）成分の処理 ---
        # 聴感変化を体感しやすいよう、オイルの不完全な損失成分(あえて実効的な120Ω)をシミュレート
        R_esr = 120.0 if has_esr else 0.0

        # 実効容量の計算と表示更新
        self.cap_label.config(text=f"実効容量: {actual_cap_uf:.4f} μF")
        self.param_label.config(text=f"[PUスペック]\n直流抵抗: {pu_param['R_pu']/1000:.1f} kΩ\nインダクタンス: {pu_param['L_pu']:.1f} H")

        # グラフエリアのリセット（重ね合わせ描画のため）
        self.ax.clear()
        self.ax.set_xscale('log')
        self.ax.set_xlim(40, 15000)
        self.ax.set_ylim(-40, 5)
        self.ax.grid(True, which="both", ls="--", color='0.8')
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Gain (dB)')

        # 1. 保存された過去の波形をすべてプロット
        for saved in self.saved_curves:
            self.ax.plot(f, saved["gain"], label=saved["label"], alpha=0.5, linestyle="--")

        # 2. 現在操作中のリアルタイム波形を太線でプロット
        current_gain = GuitarEngineV3.calculate(model, vol, tone, actual_cap_uf, pu_param, is_b_curve, is_hypass, R_esr)
        self.ax.plot(f, current_gain, color='red', linewidth=3, label="LIVE (Current Adjusting)")
        self.ax.set_title(f'Live Graph [{model} - {curve_type}]', fontsize=11, fontweight='bold')
        self.ax.legend(loc='lower left', fontsize=8)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GuitarSimAppV3(root)
    root.mainloop()
