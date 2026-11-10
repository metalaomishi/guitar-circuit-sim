import subprocess

def get_pict_cases():
    """PICTを実行してテストケースのリストを動的に生成する"""
    # PICTコマンドを実行
    result = subprocess.run(
        ['pict', 'model.txt'], 
        capture_output=True, 
        text=True, 
        check=True, 
        encoding='utf-8' # 日本語の文字化け防止
    )

    # コマンドライン出力（標準出力）
    lines = result.stdout.strip().split('\n')

    # 1行目はヘッダー（パラメータ名）なのでこれを抽出
    headers = [h.strip() for h in lines[0].split('\t')]
    
    # 2行目以降のテストデータをリスト化
    cases = []
    for line in lines[1:]:
        cases.append([value.strip() for value in line.split('\t')])
        
    return headers, cases

# PICTからテストケースを取得
headers, cases = get_pict_cases()

print(f"ヘッダー: {headers}")
print(f"テストケース数: {len(cases)}")

for i, case in enumerate(cases):
    print(f"Case {i+1}: {case}")
