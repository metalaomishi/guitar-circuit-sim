import subprocess
import csv
import io

def load_pict_cases():
    # PICTコマンドを実行して結果を取得
    result = subprocess.run(
        ["pict", "model.txt"],
        capture_output=True,
        text=True,
        check=True, 
        encoding='utf-8' # 日本語の文字化け防止
    )
    
    # 出力されるテストケースはTSV（タブ区切り）形式なので、csvモジュールでパース
    f = io.StringIO(result.stdout)
    reader = csv.DictReader(f, delimiter='\t')
    return list(reader)

cases = load_pict_cases()

print(f"生成されたテストケース数: {len(cases)}")

for i, case in enumerate(cases):
    print(f"Case {i+1}: {case}")
