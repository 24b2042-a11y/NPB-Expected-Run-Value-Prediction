# NPB 得点期待値予測アプリ

## ディレクトリ構成

```
.
├── app.py                         # Streamlit アプリ本体
├── core.py                        # RE24・wOBA 計算ロジック
├── scraper.py                     # スポナビ PBP スクレイパー（打者ID・対戦投手つき）
├── player_scraper.py              # スポナビ選手ページ → 選手マスタ（投打・球速）
├── batter_scraper.py              # nf3 打者の対戦相手別・球場別成績
├── pitcher_scraper.py             # nf3 投手成績
├── github_sync.py                 # GitHub 自動コミット
├── requirements.txt
├── .streamlit/
│   └── secrets.toml.example       # Secrets テンプレート（push しない）
└── data/
    ├── all_batters_situational.csv
    ├── all_pitchers.csv
    ├── players.csv                # 選手マスタ（選手ID・投打・ストレート平均球速など）
    └── gamedata/                  # PBP データ（アプリから自動追加）
        └── *_details.csv          # 打者ID・投手名・投手ID・攻撃 の列を含む（2026-11〜）
```

### 打席データ × 選手マスタ

`_details.csv` の `打者ID` / `投手ID` と `players.csv` の `選手ID`（スポナビの選手ID）で結合する
（`core.attach_player_info()`）。名前ではなく ID で結合するので表記ゆれの影響を受けない。

旧形式（ID 列なし）のデータを取り直す手順（ローカル）:

```bash
python scraper.py --refetch      # 旧形式の試合だけ再取得（約 2 秒/試合）
python player_scraper.py         # 未取得の選手の投打・球速を取得（約 1.5 秒/人）
git add data/ && git commit -m "PBP に投手ID を追加、players.csv を追加" && git push
```

アプリの「設定」ページからも同じ操作ができる（件数を区切って実行可能）。

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

打者 wOBA は、対戦相手別 wOBA を選手の全体 wOBA に向けてベイズ収縮した値を使う（`core.predict_batter_woba`）。

```
打者 wOBA = (PA × wOBA対戦 + C × wOBA全体) / (PA + C)     C = 20（core.SHRINK_C）
PA = 打席 − 敬遠 − 犠打（wOBA の分母）
```

対戦打席が少ないほど全体 wOBA に近づき、多いほど対戦相手別 wOBA に近づく。
