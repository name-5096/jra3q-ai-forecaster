import numpy as np
import pandas as pd
import netCDF4 as nc

# ==========================================
# 1. 設定（ファイル名を合わせてください）
# ==========================================
INPUT_NC_FILE = "data.nc" # 読み込むnetCDF4ファイルの名前
OUTPUT_CSV_FILE = (
    "gemini_data.csv"  # Geminiに読み込ませる用に出力するファイル名
)


# ==========================================
# 2. netCDF4ファイルを開いて中身を確認する
# ==========================================
print("netCDF4ファイルを読み込んでいます...")
try:
    dataset = nc.Dataset(INPUT_NC_FILE, mode="r")
except Exception as e:
    print(f"ファイルが開けません。名前が正しいか確認してください: {e}")
    exit()

# ファイルの中にどんなデータ（変数）が入っているか画面に表示します
print("含まれている気象データ一覧:", list(dataset.variables.keys()))


# ==========================================
# 3. データをGeminiが読める表形式（CSV）に変換
# ==========================================
print("Gemini用のデータに変換中...")

# 一般的なnetCDF4に含まれる「時間」「緯度」「経度」を取得
# （※ファイルによって英語名が 'time', 'lat', 'lon' など異なる場合があります）
time_var = next((v for v in dataset.variables if "time" in v.lower()), None)
lat_var = next((v for v in dataset.variables if "lat" in v.lower()), None)
lon_var = next((v for v in dataset.variables if "lon" in v.lower()), None)

# 予測に使いたい主役の気象データ（例：気温や気圧など）を自動で1つ選びます
# 緯度・経度・時間以外のデータを探します
target_var = None
for v in dataset.variables:
    if v not in [time_var, lat_var, lon_var] and len(
        dataset.variables[v].shape
    ) >= 3:
        target_var = v
        break

if not target_var:
    print("変換できる気象データ（3次元以上）が見つかりませんでした。")
    dataset.close()
    exit()

print(f"「{target_var}」のデータを抽出します。")

# データを扱いやすい数値の塊（numpy配列）として読み込みます
data_array = dataset.variables[target_var][:]

# 【初心者向け重要設定】
# netCDF4はデータ量が膨大（数百万〜数千万マス）で、そのままGeminiに渡すと
# 容量オーバーでフリーズしたり、Geminiが処理を拒否したりします。
# そのため、今回は「最初の位置・最初の時間帯の100件」だけをサンプルとして抜き出します。
flat_data = data_array.flatten()[:100]

# 表（データフレーム）を作成
df = pd.DataFrame(
    {
        "データ番号": range(1, len(flat_data) + 1),
        f"気象データ_{target_var}": flat_data,
    }
)

# ==========================================
# 4. CSVファイルとして保存
# ==========================================
# Geminiが読み込みやすいよう、シンプルなCSVファイルとして書き出します
df.to_csv(OUTPUT_CSV_FILE, index=False, encoding="utf-8")
dataset.close()

print(f"完了しました！「{OUTPUT_CSV_FILE}」というファイルが作られました。")
