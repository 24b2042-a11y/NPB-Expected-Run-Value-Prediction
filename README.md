# NPB 得点期待値予測アプリ

## ディレクトリ構成

```
.
├── app.py                         # Streamlit アプリ本体
├── core.py                        # RE24・wOBA 計算ロジック
├── scraper.py                     # スポナビ PBP スクレイパー
├── github_sync.py                 # GitHub 自動コミット
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example       # Secrets テンプレート（push しない）
└── data/
    ├── all_batters_situational.csv
    └── gamedata/                  # PBP データ（アプリから自動追加）
        └── *_details.csv
```

## Streamlit Community Cloud へのデプロイ

### 1. GitHub に push

```bash
git init && git add . && git commit -m "initial"
git remote add origin https://github.com/<your-name>/baseball-re24.git
git push -u origin main
```

### 2. share.streamlit.io でデプロイ

New app → リポジトリ選択 → Main file: `app.py` → Deploy

### 3. Secrets を設定

Streamlit Cloud の **Settings > Secrets** に貼り付け：

```toml
[github]
token     = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
repo_name = "your-username/baseball-re24"
branch    = "main"
```

> GitHub Personal Access Token は `repo` スコープが必要です。  
> https://github.com/settings/tokens で発行してください。

## アプリの使い方

### 「試合データ取得・更新」タブ

| 設定項目 | 説明 |
|---------|------|
| 開始 ID | スポナビの試合 ID（例: 2021038624） |
| 取得件数 | 連番で何試合分取得するか |
| リクエスト間隔 | サーバー負荷軽減のための待機秒数 |

取得ボタンを押すと：
1. スポナビから PBP を取得
2. `data/` に CSV を保存し GitHub にコミット
3. RE24 キャッシュをクリアして自動再計算

### 「得点期待値予測」タブ

- サイドバーで打者・対戦相手・アウト・ランナー状態を選択
- RE24 ヒートマップで全 24 場面の基礎期待得点を確認
- 選択場面の補正後期待得点をリアルタイム表示

## 補正ロジック

```
補正後期待得点 = 基礎 RE24 × (打者 wOBA / リーグ平均 wOBA)
```
