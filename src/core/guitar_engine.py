# src/core/guitar_engine.py
import numpy as np

# 周波数軸の設定 (20Hz 〜 20kHz)
f = np.logspace(1, 4.3, 300) 
w = 2 * np.pi * f
s = 1j * w # ラプラス変換の複素周波数領域での表現

# ==========================================
# 物理シミュレーション・エンジン
# ==========================================
class GuitarEngine:
    @staticmethod
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
        gain = 20 * np.log10(np.abs(V_output))  # dB換算のゲイン

        return {"frequency": f, "gain": gain}
