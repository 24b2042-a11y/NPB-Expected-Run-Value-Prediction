# NPB 得点期待値予測アプリ

PBP（打席イベント）データから計算した **RE24 行列** と打者 **wOBA** を組み合わせて、
場面（アウト × ランナー状態）ごとの期待得点を可視化するアプリです。

---

## ディレクトリ構成

```
.
├── app.py                        # Streamlit アプリ本体
├── core.py                       # 計算ロジック（RE24・wOBA）
├── requirements.txt
└── data/
    ├── all_batters_situational.csv   # 打者状況別データ
    ├── 2021038624_広島東洋カープvs.中日ドラゴンズ_details.csv
    ├── 2021038625_...._details.csv
    └── ...  ← Google Drive からエクスポートした _details.csv をすべて置く
```

---

## データの準備

Google Drive から `_details.csv` をすべてダウンロードして `data/` に配置してください。
Colab でまとめてダウンロードする場合:

```python
import shutil, glob
files = glob.glob('/content/drive/MyDrive/課題解決2026前期-野球/野球データ/年間試合データ/*_details.csv')
for f in files:
    shutil.copy(f, '/content/drive/MyDrive/baseball_app/data/')
```

---

## ローカルで起動する場合

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Streamlit Community Cloud にデプロイする手順

1. このリポジトリを GitHub に push する
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/<your-name>/baseball-re24.git
   git push -u origin main
   ```

2. [share.streamlit.io](https://share.streamlit.io) にアクセスして GitHub アカウントでログイン

3. **New app** → リポジトリ・ブランチ・`app.py` を選択して **Deploy**

4. 数分でアプリが公開される

> **注意**: `data/` フォルダ内の CSV も GitHub に push が必要です。  
> ファイルサイズが大きい場合は `.gitattributes` で Git LFS を使うか、  
> `st.file_uploader` でアップロード方式に変更してください。

---

## 機能

| 機能 | 説明 |
|------|------|
| RE24 ヒートマップ | 24 マスの期待得点を色付きで表示。選択中の場面を赤枠でハイライト |
| 1 場面予測 | 打者・対戦相手・アウト・ランナーを指定して基礎 RE24 と補正後期待得点を表示 |
| 24 場面一覧 | 指定打者×対戦相手の全組み合わせをグラフ・テーブルで表示 |
| CSV ダウンロード | 24 場面テーブルを CSV でエクスポート |

---

## 補正ロジック

```
補正後期待得点 = 基礎RE24 × (打者wOBA / リーグ平均wOBA)
```

- 打者の wOBA は `all_batters_situational.csv` の対戦相手別成績から計算
- 対戦打席数 < 10 の場合はリーグ平均 wOBA で代替
