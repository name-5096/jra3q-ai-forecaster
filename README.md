# jra3q-ai-forecaster

[English](#english) | [日本語](#日本語)

**Live Demo / 稼働中のウェブアプリ**: [https://streamlit.app](https://streamlit.app)

---

## English

An open-source data integration tool designed for severe weather research. It extracts 3D vertical atmospheric profiles from the Japan Meteorological Agency's 3rd Japanese Reanalysis (JRA-3Q) dataset to enable objective forecasting by Large Language Models (LLMs) and AI networks.

### Key Concept: Bias-Free AI Forecasting
Traditional AI prompts often introduce cognitive bias by including location names (e.g., "Kumamoto") or event labels (e.g., "Heavy Rain"). This tool is engineered to extract pure thermodynamic and dynamical numerical arrays. By feeding raw multi-level parameters into AI models via a flat prompt, it evaluates the AI's pure capability to predict self-organizing convective systems (like linear rainbands) based entirely on physical laws.

### Features
- Multi-Variable Fusion: Automatically cross-references and merges Temperature (tmp), Specific Humidity (spfh), Zonal Wind (ugrd), and Meridional Wind (vgrd) from separate netCDF4 files.
- True Vertical Profiling: Correctly maps 1D slices from surface level (1000hPa) up to the mid-troposphere (300hPa) for specific coordinates without data flattening errors.
- Convective Instability Detection: Visualizes the exact atmospheric profile where high-moisture lower layers meet cold, dry upper layers—capturing the explosive potential before a severe storm.

### Prerequisites
Install the required packages in your local environment:
```bash
pip install netCDF4 pandas numpy
```

### Dataset Layout
Download the JRA-3Q 1.25-degree isobaric analysis fields (anl_p125) for your target date and rename the files to match the layout below in the same directory:
- tmp.nc (Temperature)
- spfh.nc (Specific Humidity)
- ugrd.nc (U-component of wind)
- vgrd.nc (V-component of wind)

### Usage
Run the script to generate the combined payload:
```bash
python converter.py
```
Copy the clean console matrix output and pass it to your AI model (e.g., Gemini, GPT) with a flat prompt requesting an objective weather risk assessment.

### License
This project is licensed under the Apache License 2.0. You are free to use, modify, and distribute this software, provided that proper attribution and copyright notices are maintained.

---

## 日本語

顕著現象（集中豪雨・線状降水帯など）の予測研究を目的としたオープンソースのデータ統合ツールです。気象庁の第3次長期再解析データ（JRA-3Q）から大気の鉛直構造プロファイルを正確に抽出し、大規模言語モデル（LLM）をはじめとするAIモデルに完全客観的な予報・推論を実行させることができます。

### 研究の核心：誘導（バイアス）ゼロの気象予測
従来のAIプロンプトは、「熊本」「豪雨の前日」といった地名や災害のキーワードを含めることで、AIに強い先入観（認知バイアス）を与えてしまいがちでした。
本ツールは、地名などの情報を一切伏せ、純粋な大気熱力学・力学の数値配列のみを抽出します。生の立体データをフラットな状態でAIに読ませることで、AIが物理法則（対流不安定や鉛直風シアー）のみから「翌日に線状降水帯などの爆発的な積乱雲の発達が起きるか」を自発的に見抜けるかを検証・研究するための仕様となっています。

### 特徴
- 4大気象因子の完全統合: バラバラのファイルとして配信されるネットCDFファイル（気温・比湿・東西風・南北風）から、同一時刻・同一地点のデータを自動で吸い上げて1つのマトリックスに結合します。
- 高精度な鉛直プロファイル: 地上付近（1000hPa）から上空（300hPa）までの大気の縦方向の断面構造を、特定の観測座標にピンポイントで絞り込んで抽出します。
- 大気不安定度の可視化: 「下層が猛烈に高温高湿、上空が冷たく乾燥している」という、大雨発生直前の爆発的な潜在エネルギー（SSIやCAPEの蓄積）のシグナルをデータとして浮き彫りにします。

### 事前準備
VSCodeのターミナル等で以下のコマンドを実行し、必要なパッケージをインストールしてください：
```bash
pip install netCDF4 pandas numpy
```

### ファイルの配置
JRA-3Qの1.25度格子気圧面解析値（anl_p125）からダウンロードした4つのファイルを、本スクリプトと同じフォルダに配置し、以下のようにファイル名を短く変更してください：
- tmp.nc (気温)
- spfh.nc (比湿 / 水蒸気量)
- ugrd.nc (東西風速 / U風)
- vgrd.nc (南北風速 / V風)

### 使い方
ターミナルでプログラムを実行します：
```bash
python converter.py
```
画面に出力されたテキストの表（高度ごとの数値データ）を丸ごとコピーし、外部のAIに貼り付け、「提示した大気鉛直構造データから、翌日にこの地点周辺で起こり得る天気の崩れを客観的に予報してください」という趣旨のフラットな質問を入力して実行します。

### ライセンス
本プロジェクトは Apache License 2.0 のもとで公開されています。コードの改変や再利用は自由ですが、利用・配布の際は著作権表示および出典（クレジット）の明記が法的に義務付けられます。





### Contributing / 貢献について
Contributions, bug reports, and Pull Requests are highly welcome! If you find ways to optimize the netCDF4 parsing speed or improve the LLM prompt matrix structure, feel free to open an issue or submit a PR.
（バグ報告やプルリクエストは大歓迎です！ネットCDFの処理高速化や、LLM向けマトリックスの構造改善など、お気軽にPRをお寄せください。）
