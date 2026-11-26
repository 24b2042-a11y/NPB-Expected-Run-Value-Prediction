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

サイドバーでページを切り替える。

### 🔮 メイン（予測）
- 打者・対戦相手・アウト・ランナー状態を選ぶと、補正後期待得点を表示
- RE24 の表（ヒートマップ）、状況別の実測打率、全 24 場面の一覧、打者の個人成績、投手成績
- RE24 はレギュラーシーズンの試合だけで計算する（オールスター・ポストシーズンは `core.NON_REGULAR_IDS` で除外）

### 👥 選手成績
打者・投手の成績、今シーズンの成績一覧。

### ⚙️ 設定（データ更新）
| 項目 | 内容 |
|---|---|
| 試合データ取得・更新 | スポナビ一球速報を「取得済みの最大の試合ID＋1」から連番で取得（オールスターの ID は開始位置の計算から除く）。「試合終了」「コールド」がなく打席数の少ない試合は取り直す |
| 既存試合の再取得 | 打者ID・投手ID の列がない旧形式の試合を取り直す |
| 選手マスタの更新 | 投打・球速を `data/players.csv` に保存 |
| 打者・投手の成績更新 | nf3 から取得。12球団そろっていない等の問題があれば保存しない |

## 手元（アプリを使わない）でのデータ更新

```bash
python scraper.py --refetch                     # 旧形式の試合を新形式で取り直す
python scraper.py --ids 2021039474-2021039495   # 指定した試合ID（ポストシーズンなど）を取得
python player_scraper.py --from-teams           # 12球団の成績ページに載る選手の投打・球速
python batter_scraper.py                        # nf3 の打者成績（12球団そろわなければ保存しない）
```

取得後は `git add data/ && git commit && git push`。

## 検証（validate.py）

レギュラーシーズンで学習し、ポストシーズン（CS＋日本シリーズ 16試合）でテストする。

```bash
python validate.py                          # 学習=シーズン、テスト=ポストシーズン
python validate.py --split season-tail      # 補助: シーズン内の後ろ20%をテスト
```

- 実測値: その打席以降のイニング得点（runs_after）と、打席の得点価値 ΔRE の両方で評価
- 比較: RE24 のみ／乗法型（アプリの式）／加法型（RE24 + k(wOBA − リーグ平均)）× 打者 wOBA の作り方
- 結果は `results/` に CSV で出力（git には入れない）

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
