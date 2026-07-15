"""
core.py — RE24 計算・予測ロジック（Streamlit / Colab 共通）- バグ修正版
"""
import os
import re
import glob
import unicodedata
import numpy as np
import pandas as pd

RUNNER_MAP = {
    '':        0, '走者なし': 0,
    '1塁':     1, '2塁':     2, '3塁':     3,
    '1,2塁':   4, '1,3塁':   5, '2,3塁':   6,
    '満塁':    7,
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
# 本文テキストから打席結果を判定するためのキーワードパターン
# ============================================================
RESULT_PATTERNS = [
    ('HR',  re.compile(r'本塁打|ホームラン')),
    ('3B',  re.compile(r'三塁打|スリーベース')),
    ('2B',  re.compile(r'二塁打|ツーベース')),
    ('IBB', re.compile(r'敬遠')),
    ('BB',  re.compile(r'四球|フォアボール')),
    ('HBP', re.compile(r'死球')),
    ('SO',  re.compile(r'三振')),
    ('1B',  re.compile(r'ヒット|安打')),
    ('SF',  re.compile(r'犠飛|犠牲フライ')),
    ('SAC', re.compile(r'バント')),
    ('OUT', re.compile(r'ゴロ|フライ|ライナー|失策|エラー|併殺')),
]

# 安打として扱う結果コード
HIT_CODES    = {'1B', '2B', '3B', 'HR'}
# 打数（AB）に含めない結果コード（四球・死球・敬遠・犠打・犠飛）
NON_AB_CODES = {'BB', 'IBB', 'HBP', 'SAC', 'SF'}


def classify_result(text) -> str:
    """本文テキストから打席結果コードを判定する"""
    s = str(text) if text is not None else ''
    for code, pattern in RESULT_PATTERNS:
        if pattern.search(s):
            return code
    return 'UNKNOWN'


def _opponent_team_raw(row) -> str | None:
    """打者側から見た対戦（相手）球団の生の名前を返す"""
    if '裏' in str(row.get('イニング', '')):
        return row.get('away_team_raw')
    return row.get('home_team_raw')


def add_result_columns(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    df_runs（calc_runs_after の出力）に、打席結果・安打判定・打数対象判定・
    対戦球団（正規化済み）の列を追加して返す。
    """
    df = df_runs.copy()
    df['結果']   = df['本文'].apply(classify_result)
    df['is_hit'] = df['結果'].isin(HIT_CODES)
    df['is_ab']  = (~df['結果'].isin(NON_AB_CODES)) & (df['結果'] != 'UNKNOWN')
    df['対戦球団_raw'] = df.apply(_opponent_team_raw, axis=1)
    df['対戦球団']     = df['対戦球団_raw'].apply(normalize_team_name)
    return df


# ============================================================
# Step 1: DataFrame リストを結合して型を整える
# ============================================================
def concat_details(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['アウト']   = pd.to_numeric(all_df['アウト'],   errors='coerce').fillna(0).astype(int)
    all_df['打席順']  = pd.to_numeric(all_df['打席順'],  errors='coerce').fillna(0).astype(int)
    all_df['打順']    = pd.to_numeric(all_df['打順'],    errors='coerce')
    all_df['ランナー'] = all_df['ランナー'].fillna('')
    return all_df


# ============================================================
# Step 2: イニング順の正しいソート ＆ 残り得点の計算（バグ修正済）
# ============================================================
def sort_by_inning_correctly(df: pd.DataFrame) -> pd.DataFrame:
    """イニング文字列（10回裏が2回表より前に来る辞書順バグ）を、正しい時系列に並び替える"""
    def parse_inning_str(s):
        s = str(s)
        # イニングの数字を抽出
        num_m = re.search(r'\d+', s)
        num = int(num_m.group()) if num_m else 0
        # 「表」なら0、「裏」なら1とし、表裏順も正しく揃える
        side = 0 if '表' in s else 1
        return (num, side)
    
    df['_temp_sort'] = df['イニング'].apply(parse_inning_str)
    df = df.sort_values(['game_id', '_temp_sort', '打席順']).reset_index(drop=True)
    return df.drop(columns=['_temp_sort'])


def calc_runs_after(all_df: pd.DataFrame) -> pd.DataFrame:
    """イニング内得点差分計算ロジック（イニング独立処理で相手チーム得点の混入を防止）"""
    if all_df.empty:
        return all_df

    df = all_df.copy()
    # 1. 完璧なイニング順に並び替え
    df = sort_by_inning_correctly(df)

    # 2. 得点のパース
    scores = []
    for body in df['本文'].fillna(''):
        m = RE_SCORE.search(body)
        if m:
            scores.append(int(m.group(1)) + int(m.group(2)))
        else:
            scores.append(None)
    df['total_score'] = scores

    # 3. ゲームごとのパースエラーの穴埋めと単調増加補正
    df['total_score'] = df.groupby('game_id')['total_score'].ffill().bfill().fillna(0).astype(int)
    df['total_score'] = df.groupby('game_id')['total_score'].cummax()

    # 4. 【修正点】イニングごとに diff() を取ることでイニングまたぎ・ゲーム初期スコア誤算入を完璧に防止
    df['runs_on_play'] = df.groupby(['game_id', 'イニング'])['total_score'].diff().fillna(0).astype(int)
    df['runs_on_play'] = df['runs_on_play'].clip(lower=0)

    # 5. 各打席開始時点からイニング終了までの総残り得点(runs_after)を算出
    df['runs_after'] = df.groupby(['game_id', 'イニング'])['runs_on_play'].transform(lambda x: x[::-1].cumsum()[::-1])

    return df


# ============================================================
# 過去成績の読み込み
# ============================================================
def load_career_stats(stats_dir: str) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(stats_dir, "stats_*.csv"))
    dfs = []
    for f in all_files:
        m = re.search(r'stats_(\d+)\.csv', os.path.basename(f))
        if m:
            year = int(m.group(1))
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                df['年度'] = year
                dfs.append(df)
            except Exception:
                continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ============================================================
# 球団平均得点の算出用
# ============================================================
def build_team_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    results = []
    for game_id, grp in df_runs.groupby('game_id'):
        grp = grp.sort_values(['イニング', '打席順'])
        for _, row in grp.iloc[::-1].iterrows():
            body = str(row.get('本文', ''))
            m = RE_SCORE.search(body)
            if m:
                score_left = int(m.group(1))
                score_right = int(m.group(2))
                m_teams = re.search(r'(\S+)\s+(\d+)-(\d+)\s+(\S+)', body)
                if m_teams:
                    team_left = normalize_team_name(m_teams.group(1))
                    team_right = normalize_team_name(m_teams.group(4))
                    results.append({'game_id': game_id, 'team': team_left, 'runs': score_left})
                    results.append({'game_id': game_id, 'team': team_right, 'runs': score_right})
                    break
    if results:
        df_res = pd.DataFrame(results)
        return df_res.drop_duplicates(subset=['game_id', 'team'])
    return pd.DataFrame(columns=['game_id', 'team', 'runs'])


def get_team_avg_runs(df_team_runs: pd.DataFrame, opponent: str) -> tuple[float | None, int]:
    if df_team_runs.empty:
        return None, 0
    df_opp = df_team_runs[df_team_runs['team'] == opponent]
    if df_opp.empty:
        return None, 0
    return float(df_opp['runs'].mean()), int(df_opp['game_id'].nunique())


# ============================================================
# 選手個人成績
# ============================================================
def get_player_career(df_career: pd.DataFrame, batter: str) -> pd.DataFrame:
    if df_career.empty:
        return pd.DataFrame()
    return df_career[df_career['選手名'] == batter].copy()


def get_player_current_team(df_career: pd.DataFrame, batter: str) -> str | None:
    if df_career.empty:
        return None
    df_b = df_career[df_career['選手名'] == batter]
    if df_b.empty:
        return None
    latest_row = df_b.sort_values('年度', ascending=False).iloc[0]
    return latest_row.get('所属球団')


# ============================================================
# モデル構築（メインエントリーポイント）
# ============================================================
def build_model_from_dfs(dfs: list[pd.DataFrame], batter_df: pd.DataFrame | None, stats_dir: str):
    """app.pyの load_model から呼び出されるモデルデータビルド"""
    if not dfs:
        empty_re24 = pd.DataFrame(0.0, index=[0, 1, 2], columns=RUNNER_LABEL)
        empty_counts = pd.DataFrame(0, index=[0, 1, 2], columns=RUNNER_LABEL)
        return (empty_re24, empty_counts, pd.DataFrame(), 0.315, 0, 0,
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 1. 結合・イニング別残り得点の計算
    all_df = concat_details(dfs)
    df_runs = calc_runs_after(all_df)
    df_runs = add_result_columns(df_runs)

    df_runs['選手名'] = df_runs.get('打者', df_runs.get('選手名', '')).fillna('')

    # 2. 【修正点】RE24 計算時に「投手交代」「代打」などの打席を伴わないノイズイベントを完全排除
    df_runs_for_re24 = df_runs[(df_runs['結果'] != 'UNKNOWN') & (df_runs['アウト'].isin([0, 1, 2]))]

    re24_pivot = df_runs_for_re24.pivot_table(index='アウト', columns='ランナー', values='runs_after', aggfunc='mean')
    counts_pivot = df_runs_for_re24.pivot_table(index='アウト', columns='ランナー', values='runs_after', aggfunc='count')

    re24 = pd.DataFrame(np.nan, index=[0, 1, 2], columns=RUNNER_LABEL)
    counts = pd.DataFrame(0, index=[0, 1, 2], columns=RUNNER_LABEL)
    for out in [0, 1, 2]:
        for runner in RUNNER_LABEL:
            if out in re24_pivot.index and runner in re24_pivot.columns:
                re24.loc[out, runner] = re24_pivot.loc[out, runner]
                counts.loc[out, runner] = counts_pivot.loc[out, runner]

    # 3. リーグ平均 wOBA の算出
    counts_all = df_runs['結果'].value_counts()
    woba_num = (
        0.70 * counts_all.get('BB', 0) +
        0.73 * counts_all.get('HBP', 0) +
        0.89 * counts_all.get('1B', 0) +
        1.27 * counts_all.get('2B', 0) +
        1.61 * counts_all.get('3B', 0) +
        2.10 * counts_all.get('HR', 0)
    )
    ab_sum = df_runs['is_ab'].sum()
    bb_sum = counts_all.get('BB', 0)
    hbp_sum = counts_all.get('HBP', 0)
    sf_sum = counts_all.get('SF', 0)
    woba_den = ab_sum + bb_sum + hbp_sum + sf_sum
    league_avg = woba_num / woba_den if woba_den > 0 else 0.315

    # 4. wOBA マッピング
    df_woba = pd.DataFrame(columns=['選手名', '区分名', '試合', '打席', 'wOBA'])
    if batter_df is not None and not batter_df.empty:
        col_map = {}
        for col in batter_df.columns:
            if '選手' in col: col_map[col] = '選手名'
            elif '区分' in col or '対戦' in col or '球場' in col: col_map[col] = '区分名'
            elif '試合' in col: col_map[col] = '試合'
            elif '打席' in col: col_map[col] = '打席'
            elif 'wOBA' in col or 'woba' in col: col_map[col] = 'wOBA'

        df_woba = batter_df.rename(columns=col_map)
        for col in ['選手名', '区分名', '試合', '打席', 'wOBA']:
            if col not in df_woba.columns:
                if col in ['試合', '打席']: df_woba[col] = 0
                elif col == 'wOBA': df_woba[col] = league_avg
                else: df_woba[col] = ''
        df_woba = df_woba[['選手名', '区分名', '試合', '打席', 'wOBA']].copy()

    n_games = all_df['game_id'].nunique()
    n_pa = len(all_df)

    df_career = load_career_stats(stats_dir)
    df_team_runs = build_team_runs(df_runs)

    # 5. 球団別対戦成績 (UNKNOWN を除いた有効打撃プレーのみ対象)
    df_runs_valid = df_runs_for_re24[df_runs_for_re24['対戦球団'].notna() & (df_runs_for_re24['選手名'] != '')]
    if not df_runs_valid.empty:
        stats_rows = []
        for (player, opp), grp in df_runs_valid.groupby(['選手名', '対戦球団']):
            pa = len(grp)
            ab = grp['is_ab'].sum()
            h = grp['is_hit'].sum()
            avg = h / ab if ab > 0 else np.nan
            exp_runs = grp['runs_after'].mean()
            if pa < 5:
                avg = np.nan
                exp_runs = np.nan
            stats_rows.append({
                '選手名': player,
                '対戦球団': opp,
                '打席': pa,
                '打数': ab,
                '安打': h,
                '打率': avg,
                '得点期待値': exp_runs
            })
        df_team_batting_stats = pd.DataFrame(stats_rows)
    else:
        df_team_batting_stats = pd.DataFrame(columns=['選手名', '対戦球団', '打席', '打数', '安打', '打率', '得点期待値'])

    # 6. 状況別成績 (こちらも UNKNOWN を除いた有効データを反映)
    situ_rows = []
    for out in [0, 1, 2]:
        for runner in RUNNER_LABEL:
            grp = df_runs_for_re24[(df_runs_for_re24['アウト'] == out) & (df_runs_for_re24['ランナー'] == runner)]
            pa = len(grp)
            ab = grp['is_ab'].sum() if pa > 0 else 0
            h = grp['is_hit'].sum() if pa > 0 else 0
            avg = h / ab if ab > 0 else np.nan
            exp_runs = grp['runs_after'].mean() if pa > 0 else np.nan
            if pa < 5:
                avg = np.nan
                exp_runs = np.nan
            situ_rows.append({
                'アウト': f"{out}アウト",
                'ランナー': runner,
                '打席': pa,
                '打数': ab,
                '安打': h,
                '打率': avg,
                '得点期待値': exp_runs
            })
    df_situational_stats = pd.DataFrame(situ_rows)

    return (re24, counts, df_woba, league_avg, n_games, n_pa, df_career,
            df_team_runs, df_team_batting_stats, df_situational_stats)


# ============================================================
# 予測ロジック
# ============================================================
def predict_one(re24, df_woba, league_avg, batter, opponent, out, runner):
    """特定の場面の得点期待値を予測する（ベイズ収縮補正入り）"""
    try:
        val = re24.loc[out, runner]
        base_re = float(val) if pd.notna(val) else None
    except Exception:
        base_re = None

    df_b = df_woba[(df_woba['選手名'] == batter) & (df_woba['区分名'] == opponent)]
    if not df_b.empty:
        pa = int(df_b.iloc[0]['打席'])
        games = int(df_b.iloc[0].get('試合', 0))
        woba_vs = float(df_b.iloc[0]['wOBA'])
    else:
        pa = 0
        games = 0
        woba_vs = league_avg

    df_all_b = df_woba[df_woba['選手名'] == batter]
    if not df_all_b.empty and df_all_b['打席'].sum() > 0:
        woba_overall = (df_all_b['wOBA'] * df_all_b['打席']).sum() / df_all_b['打席'].sum()
    else:
        woba_overall = league_avg

    C = 20
    woba_pred = (pa * woba_vs + C * woba_overall) / (pa + C) if (pa + C) > 0 else league_avg

    if base_re is not None and league_avg > 0:
        pred_re = base_re * (woba_pred / league_avg)
    else:
        pred_re = None

    note = None
    if pa < 10:
        note = f"対 {opponent} の打席数（{pa}打席）が少ないため、全体成績（wOBA: {woba_overall:.3f}）を加重して予測を補正しています。"

    return {
        '基礎RE24': round(base_re, 3) if base_re is not None else None,
        '打者wOBA': round(woba_pred, 3),
        '補正後期待得点': round(pred_re, 3) if pred_re is not None else None,
        '対戦打席数': pa,
        '対戦試合数': games,
        'note': note
    }


def predict_all(re24, df_woba, league_avg, batter, opponent):
    """24場面すべての予測結果をDataFrameにまとめて返す"""
    rows = []
    for out in [0, 1, 2]:
        for runner in RUNNER_LABEL:
            res = predict_one(re24, df_woba, league_avg, batter, opponent, out, runner)
            rows.append({
                'アウト': out,
                'ランナー': runner,
                '打者wOBA': res['打者wOBA'],
                '基礎RE24': res['基礎RE24'],
                '補正後期待得点': res['補正後期待得点'],
                '対戦打席数': res['対戦打席数']
            })
    return pd.DataFrame(rows)
