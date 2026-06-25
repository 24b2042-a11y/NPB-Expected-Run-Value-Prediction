"""
core.py — RE24 計算・予測ロジック（Streamlit / Colab 共通）
Google Drive / Colab 依存を完全に除去
"""
import os, re, glob
import numpy as np
import pandas as pd

# ============================================================
# 定数
# ============================================================
RUNNER_MAP = {
    '':       0, '走者なし': 0,
    '1塁':    1, '2塁':    2, '3塁':    3,
    '1,2塁':  4, '1,3塁':  5, '2,3塁':  6,
    '満塁':   7,
}
RUNNER_LABEL = ['走者なし', '1塁', '2塁', '3塁', '1,2塁', '1,3塁', '2,3塁', '満塁']

WOBA_W = dict(BB=0.70, HBP=0.73, S=0.89, D=1.27, T=1.61, HR=2.10)

RE_SCORE = re.compile(r'\S+\s+(\d+)-(\d+)\s+\S+')


# ============================================================
# Step 1: _details.csv を全試合分読み込む
# ============================================================
def load_all_details(details_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(details_dir, '*_details.csv'))
    if not files:
        raise FileNotFoundError(f"_details.csv が見つかりません: {details_dir}")
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        base   = os.path.basename(f)
        parts  = base.split('_', 1)
        df['game_id'] = parts[0]
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['アウト']   = pd.to_numeric(all_df['アウト'],   errors='coerce').fillna(0).astype(int)
    all_df['打席順']  = pd.to_numeric(all_df['打席順'],  errors='coerce').fillna(0).astype(int)
    all_df['打順']    = pd.to_numeric(all_df['打順'],    errors='coerce')
    all_df['ランナー'] = all_df['ランナー'].fillna('')
    return all_df, len(files)


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

        for i, (_, row) in enumerate(grp.iterrows()):
            results.append({
                'game_id':    game_id,
                'イニング':   inning,
                '打席順':     row['打席順'],
                '選手名':     row['選手名'],
                'アウト':     row['アウト'],
                'ランナー':   row['ランナー'],
                '本文':       row['本文'],
                'runs_after': int(suffix[i]),
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
    return df[['選手名', '選手名_key', '区分名', '球団', '打席', 'wOBA']].copy(), float(league_avg)


# ============================================================
# Step 5: 1場面の期待得点を予測
# ============================================================
def predict_one(re24, df_woba, league_avg,
                batter_name: str, opponent: str,
                out: int, runner: str) -> dict:
    runner_idx = RUNNER_MAP.get(runner, 0)
    base_re    = re24.iloc[out, runner_idx]

    key = re.sub(r'[\s\u3000]', '', batter_name)
    hit = df_woba[(df_woba['選手名_key'] == key) & (df_woba['区分名'] == opponent)]

    if len(hit) > 0 and not pd.isna(hit['wOBA'].values[0]):
        batter_woba = float(hit['wOBA'].values[0])
        pa          = int(hit['打席'].values[0])
        data_note   = ''
    else:
        batter_woba = league_avg
        pa          = 0
        data_note   = f'※ {batter_name} vs {opponent} のデータなし → リーグ平均を使用'

    if not np.isnan(base_re) and league_avg > 0:
        adj_re = base_re * (batter_woba / league_avg)
    else:
        adj_re = np.nan

    return {
        'アウト':         out,
        'ランナー':       runner if runner else '走者なし',
        '打者wOBA':       round(batter_woba, 3),
        '対戦打席数':     pa,
        '基礎RE24':       round(float(base_re), 3) if not np.isnan(base_re) else None,
        '補正後期待得点': round(float(adj_re),  3) if not np.isnan(adj_re)  else None,
        'note':           data_note,
    }


# ============================================================
# Step 6: 全 24 場面テーブル
# ============================================================
def predict_all(re24, df_woba, league_avg,
                batter_name: str, opponent: str) -> pd.DataFrame:
    rows = []
    for o in range(3):
        for r in RUNNER_LABEL:
            rows.append(predict_one(re24, df_woba, league_avg,
                                    batter_name, opponent, o, r))
    return pd.DataFrame(rows)


# ============================================================
# モデル一括構築（キャッシュ用）
# ============================================================
def build_model(details_dir: str, batter_csv: str):
    # データなし → RE24はNaN埋めで返し、予測タブ側で案内する
    os.makedirs(details_dir, exist_ok=True)
    files = glob.glob(os.path.join(details_dir, '*_details.csv'))
    if files:
        all_df, n_games = load_all_details(details_dir)
        df_runs         = calc_runs_after(all_df)
        re24, counts    = build_re24(df_runs)
        n_pa            = len(all_df)
    else:
        re24   = pd.DataFrame(
            np.full((3, 8), np.nan),
            index=pd.Index([0, 1, 2], name='アウト'),
            columns=pd.Index(RUNNER_LABEL, name='ランナー状態'),
        )
        counts  = np.zeros((3, 8), dtype=int)
        n_games = 0
        n_pa    = 0

    df_woba, league_avg = build_batter_woba(batter_csv)
    return re24, counts, df_woba, league_avg, n_games, n_pa
