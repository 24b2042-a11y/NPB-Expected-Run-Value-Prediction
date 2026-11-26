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

## 補正ロジック（2026-11-26〜 打者タイプ × 投手タイプの相性モデル）

```
補正後期待得点 = RE24[アウト, 走者] + k × (w − リーグ平均wOBA)
w = 打者wOBA + (投手の被wOBA − リーグ平均) + 相性[打者タイプ, 投手タイプ]
```

- すべてレギュラーシーズンの一球速報から計算（`core.build_type_model`）
- 打者wOBA・投手の被wOBA は打席の少ない選手をリーグ平均へ収縮（C = 100 打席）
- k は「打席の wOBA の値」と「打席の得点価値 ΔRE」の関係から推定（約 0.76）
- 打者タイプ: 打席（右/左。両打は投手の逆側）× 長打型/巧打型（ISO が中央値以上で長打型）
- 投手タイプ: 投げ腕（右/左）× 速球派/技巧派（速球の平均球速と奪三振率の z 値の平均が 0 以上で速球派）
- 相性: 4 × 4 = 16 マス。打者・投手の実力から期待される wOBA との差を、打席の少ないマスほど 0 に寄せる（C = 300）
- アプリでは対戦投手を1人選ぶか、「指定しない」で相手チームの投手全体（登板の多さで重み付け）

以前の式（RE24 × 打者wOBA / リーグ平均、対戦相手別 wOBA を C = 20 で収縮）は、検証で RE24 だけより悪かったため置き換えた。

## 調子（好調・普通・不調）

- 判定: 直近の試合（打者 10 試合・投手 5 登板）の wOBA（投手は被 wOBA）とシーズンの実力の差を z 値で見る（`core.build_form_table`）。
  z = (直近の wOBA 分子 − 直近の打席 × シーズン wOBA) / (0.50 × √直近の打席)。打者は z ≥ 1 で好調・z ≤ −1 で不調（投手は逆向き）。15 打席未満は普通
- 効き方: 好調で打者 wOBA ×1.10、不調で ×0.90（投手は被 wOBA に逆向き）。`core.FORM_RATE`
- アプリの既定は全員「普通」。判定結果は参考として表示し、選んだときだけ使う
- 注意: 2026年のデータでは、直近の調子が次の試合の成績を当てる効果は確認できなかった（参考機能）

## 試合シミュレーション（game_sim.py）

スタメン9人と先発投手を入力すると、モンテカルロ法で試合を繰り返し、勝率・予想スコア・1試合の流れを出す。

- 1打席の結果（三振・凡退・四死球・単打・二塁打・三塁打・本塁打）の確率: 打者と投手の割合（リーグ平均へ C=100 で収縮）を
  オッズ比法（一般化 log5）で合成し、打者タイプ × 投手タイプの相性の倍率と調子を掛ける
- 走者・アウトの進み方: 一球速報の「状況（24通り）× 結果」ごとの実際の遷移（併殺・進塁打・盗塁なども含む）から引く
- 投手交代: 先発は今季の先発登板での対戦打者数で降板し、その後はチームの救援陣（救援での対戦打者数で重み付け）
- 9回裏以降はサヨナラで終了、12回で同点なら引き分け
- 検証: `python validate.py --sim`（ポストシーズン16試合を実際のスタメンで再現し、得点と勝敗を比べる）

## wOBA の係数（NPB 2026 用）と条件別 wOBA

- 係数は 2026年 NPB レギュラーシーズンの一球速報から算出（`core.compute_woba_weights`。Tango の方法）
  - 各結果の得点価値 = その結果の打席の ΔRE（RE24 の変化 + 入った得点）の平均
  - 係数 = (得点価値 − 凡退の得点価値 −0.231) × wOBA スケール 1.344（リーグ平均 wOBA = リーグ出塁率 0.306 になるように調整）

| 結果 | NPB 2026（使用中） | MLB（以前） |
|---|---|---|
| 四球（敬遠除く） | 0.726 | 0.70 |
| 死球 | 0.735 | 0.73 |
| 単打 | 0.889 | 0.89 |
| 二塁打 | 1.360 | 1.27 |
| 三塁打 | 1.858 | 1.61 |
| 本塁打 | 2.164 | 2.10 |

- `core.WOBA_W` で切り替え（`WOBA_W_NPB2026` / `WOBA_W_MLB`）。nf3 の「四球」は敬遠を含むので、敬遠を引いてから係数を掛ける
- アプリの「📈 条件別 wOBA」ページ: 打者（投手なら被 wOBA）を選び、期間（前半・後半・直近N試合・ポストシーズン）、
  相手の左右・タイプ、対戦球団、走者（得点圏・満塁など）、アウト、ホーム/ビジターで絞り込んで wOBA を計算。
  同じ条件のリーグ平均と、条件ごとの内訳も表示（`core.build_pa_table` / `woba_stats` / `woba_breakdown`）
- メインの予測でも条件別 wOBA を使える（`core.conditional_batter_woba`）。既定は「走者・アウト」。
  条件での wOBA を打者の全体 wOBA へ 200 打席分寄せた値を打者 wOBA に使う。
  検証では走者・アウト・得点圏・対戦球団で絞っても予測はほぼ変わらない（打席が少なく偶然の影響が大きい）
