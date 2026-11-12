エレキギター回路・物理シミュレータ

```text
guitar-circuit-sim/
├── .github/
│   └── workflows/
│       └── test.yml         # GitHub Actionsの設定ファイル
├── .gitignore               # Git管理除外ファイル
├── src/                     # プロジェクトのソースコード
│   ├── __init__.py          # Pythonパッケージ情報
│   ├── core/                # シミュレーションエンジン・計算コア
│   │   ├── __init__.py      # 
│   │   └── guitar_engine.py # 計算・回帰予測ロジック
│   ├── ui/                  # ユーザーインターフェース層
│   │   ├── __init__.py      # Pythonパッケージ情報
│   │   └── app.py           # メインUI
│   └── pickups.json         # ピックアップデータベース
├── tests/                   # テストコード用フォルダ
│   ├── __init__.py          # Pythonパッケージ情報
│   ├── conftest.py          # pytestの共通設定（必要に応じて）
│   ├── test_sim.py          # シミュレータテスト
│   └── test_ui.py           # UIテスト
├── main.py
├── README.md
└── requirements.txt         # 依存ライブラリ一覧
```
