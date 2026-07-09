"""
core.py — RE24 計算・予測ロジック（Streamlit / Colab 共通）
"""
import os, re, glob, unicodedata
import numpy as np
import pandas as pd

RUNNER_MAP = {
    '':       0, '走者なし': 0,
    '1塁':    1, '2塁':    2, '3塁':    3,
    '1,2塁':  4, '1,3塁':  5, '2,3塁':  6,
    '満塁':   7,
}
RUNNER_LABEL = ['走者なし', '1塁', '2塁', '3塁', '1,2塁', '1,3塁', '2,3塁', '満塁']
WOBA_W       = dict(BB=0.70, HBP=0.73, S=0.89, D=1.27, T=1.61, HR=2.10)
RE_SCORE     = re.compile(r'\S+\s+(\d+)-(\d+)\s+\S+')

# 球団の正式名称（カード表記）→ 状況別データの区分名に統一するマッピング
TEAM_NAME_MAP = [
    ('広島', '広島'), ('中日', '中日'), ('ロッテ', 'ロッテ'), ('西武', '西武'),
    ('オリックス', 'オリックス'), ('楽天', '楽天'), ('ソフトバンク', 'ソフトバンク'),
    ('日本ハム', '日本ハム'), ('読売', '巨人'), ('阪神', '阪神'),
    ('DeNA', 'ＤｅＮＡ'), ('ヤクルト', 'ヤクルト'),
]


def normalize_team_name(raw: str | None) -> str | None:
    """カードの長い球団名を状況別データの区分名（短縮名）に変換する"""
    if not raw:
        return None
    s = unicodedata.normalize('NFKC', raw)
    for key, target in TEAM_NAME_MAP:
        if key in s:
            return target
    return raw


# ============================================================
# Step 1: DataFrame リストを結合して型を整える
# ============================================================
def concat_details(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['アウト']   = pd.to_numeric(all_df['アウト'],   errors='coerce').fillna(0).astype(int)
    all_df['打席順']  = pd.to_numeric(all_df['打席順'],  errors='coerce').fillna(0).astype(int)
    all_df['打順']    = pd.to_numeric(all_df['打順'],    errors='coerce')
    all_df['ランナー'] = all_df['ランナー'].fillna('')
    return all_df


# ============================================================
# Step 2: イニング残り得点を付与
# ============================================================
def calc_runs_after(all_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for (game_id, inning), grp in all_df.groupby(['game_id', 'イニング']):
        grp = grp.sort_values('打席順').reset_index(drop=True)

        score_after, last_score = [None] * len(grp), None
        for i, body in enumerate(grp['本文'].fillna('')):
            m = RE_SCORE.search(str(body))
            if m:
                last_score = int(m.group(1)) + int(m.group(2))
            score_after[i] = last_score

        delta, prev = [], 0
        for s in score_after:
            cur = s if s is not None else prev
            delta.append(cur - prev)
            prev = cur

        suffix = np.cumsum(delta[::-1])[::-1]

        home_raw = grp['home_team_raw'].iloc[0] if 'home_team_raw' in grp.columns else None
        away_raw = grp['away_team_raw'].iloc[0] if 'away_team_raw' in grp.columns else None

        for i, (_, row) in enumerate(grp.iterrows()):
            results.append({
                'game_id':       game_id,
                'イニング':      inning,
                '打席順':        row['打席順'],
                '選手名':        row['選手名'],
                'アウト':        row['アウト'],
                'ランナー':      row['ランナー'],
                '本文':          row['本文'],
                'runs_after':    int(suffix[i]),
                'home_team_raw': home_raw,
                'away_team_raw': away_raw,
            })
    return pd.DataFrame(results)


# ============================================================
# Step 3: RE24 行列を集計
# ============================================================
def build_re24(df_runs: pd.DataFrame):
    df = df_runs.copy()
    df['runner_idx'] = df['ランナー'].map(RUNNER_MAP).fillna(0).astype(int)

    re24   = np.zeros((3, 8))
    counts = np.zeros((3, 8), dtype=int)

    for _, row in df.iterrows():
        o, r = int(row['アウト']), int(row['runner_idx'])
        if 0 <= o <= 2 and 0 <= r <= 7:
            re24[o, r]   += row['runs_after']
            counts[o, r] += 1

    with np.errstate(invalid='ignore'):
        re24_avg = np.where(counts >= 5, re24 / np.maximum(counts, 1), np.nan)

    df_re24 = pd.DataFrame(
        re24_avg,
        index=pd.Index([0, 1, 2], name='アウト'),
        columns=pd.Index(RUNNER_LABEL, name='ランナー状態'),
    )
    return df_re24, counts


# ============================================================
# Step 3.5: チーム別・試合別の得点を集計
# ============================================================
def build_team_game_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    df_runs（calc_runs_after の出力）から、各試合・各チームの総得点を計算する。

    アルゴリズム:
      各半イニングの「先頭打者（打席順=1）」の runs_after は、
      そのイニングの suffix sum なのでイニング総得点そのものになる。
      表イニング → アウェイチームの得点 / 裏イニング → ホームチームの得点
      これを試合ごとに合算するとその試合のチーム総得点になる。

    Returns
    -------
    DataFrame [game_id, team, runs]  team は正規化済みの短縮球団名
    """
    first = df_runs[df_runs['打席順'] == 1].copy()
    if first.empty:
        return pd.DataFrame(columns=['game_id', 'team', 'runs'])

    def _team_raw(row):
        return row['home_team_raw'] if '裏' in str(row['イニング']) else row['away_team_raw']

    first['team_raw'] = first.apply(_team_raw, axis=1)
    first['team']     = first['team_raw'].apply(normalize_team_name)

    grouped = (first.groupby(['game_id', 'team'])['runs_after']
              .sum().reset_index().rename(columns={'runs_after': 'runs'}))
    return grouped


def get_team_avg_runs(df_team_runs: pd.DataFrame, team: str) -> tuple[float | None, int]:
    """
    指定チームの1試合平均得点と集計試合数を返す。
    """
    sub = df_team_runs[df_team_runs['team'] == team]
    if sub.empty:
        return None, 0
    return float(sub['runs'].mean()), len(sub)


# ============================================================
# Step 4: 打者 wOBA テーブルを構築
# ============================================================
def build_batter_woba(batter_csv: str):
    df = pd.read_csv(batter_csv, encoding='utf-8-sig')
    df = df[df['区分種別'] == '対戦相手'].copy()
    df['選手名_key'] = df['選手名'].str.replace(r'[\s　]', '', regex=True)
    df['1B_cnt']    = df['安打'] - df['2B'] - df['3B'] - df['本塁']
    df['wOBA_num']  = (WOBA_W['BB']  * df['四球'] +
                       WOBA_W['HBP'] * df['死球'] +
                       WOBA_W['S']   * df['1B_cnt'] +
                       WOBA_W['D']   * df['2B'] +
                       WOBA_W['T']   * df['3B'] +
                       WOBA_W['HR']  * df['本塁'])
    df['wOBA_den']  = df['打席'] - df['敬遠'] - df['犠打']
    df['wOBA']      = np.where(df['wOBA_den'] >= 10,
                               df['wOBA_num'] / df['wOBA_den'], np.nan)
    league_avg      = df[df['wOBA_den'] >= 30]['wOBA'].mean()
    return df[['選手名', '選手名_key', '区分名', '球団', '試合', '打席', 'wOBA']].copy(), float(league_avg)


# ============================================================
# Step 5: 1場面の予測
# ============================================================
def predict_one(re24, df_woba, league_avg,
                batter_name: str, opponent: str,
                out: int, runner: str) -> dict:
    runner_idx = RUNNER_MAP.get(runner, 0)
    base_re    = re24.iloc[out, runner_idx]

    key = re.sub(r'[\s　]', '', batter_name)
    hit = df_woba[(df_woba['選手名_key'] == key) & (df_woba['区分名'] == opponent)]

    if len(hit) > 0 and not pd.isna(hit['wOBA'].values[0]):
        batter_woba = float(hit['wOBA'].values[0])
        pa          = int(hit['打席'].values[0])
        games       = int(hit['試合'].values[0])
        note        = ''
    else:
        batter_woba = league_avg
        pa          = 0
        games       = 0
        note        = f'※ {batter_name} vs {opponent} のデータなし → リーグ平均を使用'

    adj_re = (base_re * (batter_woba / league_avg)
              if not np.isnan(base_re) and league_avg > 0 else np.nan)

    return {
        'アウト':         out,
        'ランナー':       runner if runner else '走者なし',
        '打者wOBA':       round(batter_woba, 3),
        '対戦打席数':     pa,
        '対戦試合数':     games,
        '基礎RE24':       round(float(base_re), 3) if not np.isnan(base_re) else None,
        '補正後期待得点': round(float(adj_re),  3) if not np.isnan(adj_re)  else None,
        'note':           note,
    }


# ============================================================
# Step 6: 全 24 場面テーブル
# ============================================================
def predict_all(re24, df_woba, league_avg,
                batter_name: str, opponent: str) -> pd.DataFrame:
    return pd.DataFrame([
        predict_one(re24, df_woba, league_avg, batter_name, opponent, o, r)
        for o in range(3) for r in RUNNER_LABEL
    ])


# ============================================================
# Step 7: 選手個人の過去成績（stats_YYYY.csv）を読み込む
# ============================================================
def load_career_stats(stats_dir: str) -> pd.DataFrame:
    """
    data/2023~2025打撃データ/ 内の stats_2023.csv, stats_2024.csv, stats_2025.csv
    を全件結合して返す。ファイルが存在しない年はスキップする。

    列: player_id, 選手名, 年度, 試合, 打席, 打数, 得点, 安打, 二塁打, 三塁打,
        本塁打, 塁打, 打点, 盗塁, 盗塁刺, 四球, 死球, 三振, 併殺打,
        打率, 出塁率, 長打率, 犠打, 犠飛, 所属球団
    """
    files = sorted(glob.glob(os.path.join(stats_dir, 'stats_*.csv')))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    all_stats = pd.concat(dfs, ignore_index=True)
    all_stats['選手名_key'] = all_stats['選手名'].str.replace(r'[\s　]', '', regex=True)
    all_stats['OPS'] = all_stats['出塁率'] + all_stats['長打率']
    return all_stats


def get_player_career(df_stats: pd.DataFrame, batter_name: str) -> pd.DataFrame:
    """
    指定選手の過去成績を年度昇順で返す。
    """
    if df_stats.empty:
        return pd.DataFrame()
    key = re.sub(r'[\s　]', '', batter_name)
    result = df_stats[df_stats['選手名_key'] == key].sort_values('年度')
    return result


def get_player_current_team(df_stats: pd.DataFrame, batter_name: str) -> str | None:
    """
    最新年度の所属球団を返す。データがなければ None。
    """
    career = get_player_career(df_stats, batter_name)
    if career.empty:
        return None
    return career.iloc[-1]['所属球団']


# ============================================================
# モデル一括構築（DataFrame リストから）
# ============================================================
def build_model_from_dfs(dfs: list[pd.DataFrame], batter_csv: str, stats_dir: str | None = None):
    """
    dfs: GitHub から読み込んだ _details.csv の DataFrame リスト
    stats_dir: 過去3年成績 CSV が入っているディレクトリ（任意）
    """
    if dfs:
        all_df   = concat_details(dfs)
        df_runs  = calc_runs_after(all_df)
        re24, counts = build_re24(df_runs)
        df_team_runs = build_team_game_runs(df_runs)
        n_pa     = len(all_df)
    else:
        re24 = pd.DataFrame(
            np.full((3, 8), np.nan),
            index=pd.Index([0, 1, 2], name='アウト'),
            columns=pd.Index(RUNNER_LABEL, name='ランナー状態'),
        )
        counts       = np.zeros((3, 8), dtype=int)
        df_team_runs = pd.DataFrame(columns=['game_id', 'team', 'runs'])
        n_pa         = 0

    df_woba, league_avg = build_batter_woba(batter_csv)
    df_career = load_career_stats(stats_dir) if stats_dir else pd.DataFrame()

    return re24, counts, df_woba, league_avg, len(dfs), n_pa, df_career, df_team_runs
