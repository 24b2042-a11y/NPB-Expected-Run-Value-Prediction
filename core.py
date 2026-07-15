"""
core.py — RE24 計算・予測ロジック（対戦相手＆球場 分離拡張版）
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
    if not raw or raw in ('対戦相手', '対戦球団', '相手球団'):
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
    df_runsに、打席結果・安打判定・打数対象判定・
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
    
    # 空文字や欠損値を、「走者なし」に統一
    all_df['ランナー'] = all_df['ランナー'].fillna('走者なし').replace({
        '': '走者なし', 
        'nan': '走者なし', 
        'None': '走者なし'
    })
    return all_df


# ============================================================
# Step 2: イニング順の正しいソート ＆ 残り得点の計算
# ============================================================
def sort_by_inning_correctly(df: pd.DataFrame) -> pd.DataFrame:
    """イニング文字列を正しい時系列に並び替える"""
    def parse_inning_str(s):
        s = str(s)
        num_m = re.search(r'\d+', s)
        num = int(num_m.group()) if num_m else 0
        side = 0 if '表' in s else 1
        return (num, side)
    
    df['_temp_sort'] = df['イニング'].apply(parse_inning_str)
    df = df.sort_values(['game_id', '_temp_sort', '打席順']).reset_index(drop=True)
    return df.drop(columns=['_temp_sort'])


def calc_runs_after(all_df: pd.DataFrame) -> pd.DataFrame:
    """イニング内得点差分計算ロジック"""
    if all_df.empty:
        return all_df

    df = all_df.copy()
    df = sort_by_inning_correctly(df)

    scores = []
    for body in df['本文'].fillna(''):
        m = RE_SCORE.search(body)
        if m:
            scores.append(int(m.group(1)) + int(m.group(2)))
        else:
            scores.append(None)
    df['total_score'] = scores

    df['total_score'] = df.groupby('game_id')['total_score'].ffill().bfill().fillna(0).astype(int)
    df['total_score'] = df.groupby('game_id')['total_score'].cummax()

    df['runs_on_play'] = df.groupby(['game_id', 'イニング'])['total_score'].diff().fillna(0).astype(int)
    df['runs_on_play'] = df['runs_on_play'].clip(lower=0)

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
    if not dfs:
        empty_re24 = pd.DataFrame(0.0, index=[0, 1, 2], columns=RUNNER_LABEL)
        empty_counts = pd.DataFrame(0, index=[0, 1, 2], columns=RUNNER_LABEL)
        return (empty_re24, empty_counts, pd.DataFrame(), 0.315, 0, 0,
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    # 1. 結合・残り得点の計算
    all_df = concat_details(dfs)
    df_runs = calc_runs_after(all_df)
    df_runs = add_result_columns(df_runs)

    df_runs['選手名'] = df_runs.get('打者', df_runs.get('選手名', '')).fillna('')

    # 2. 有効打席のみのRE24用データ抽出
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

    # ============================================================
    # 4. wOBA マッピング (「区分（対戦相手/球場）」と「区分名」の明確な分離)
    # ============================================================
    df_woba = pd.DataFrame(columns=['選手名', '区分', '区分名', '試合', '打席', 'wOBA'])
    if batter_df is not None and not batter_df.empty:
        col_map = {}
        
        # [Step 4-1] 厳密マッチ
        for col in batter_df.columns:
            if col in ['選手名', '選手']:
                col_map[col] = '選手名'
            elif col == '区分':
                col_map[col] = '区分'
            elif col in ['区分名', '対戦相手', '対象', '球場名']:
                col_map[col] = '区分名'
            elif col in ['試合', '試合数']:
                col_map[col] = '試合'
            elif col in ['打席', '打席数']:
                col_map[col] = '打席'
            elif col.lower() == 'woba':
                col_map[col] = 'wOBA'

        # [Step 4-2] 部分一致フォールバック
        mapped_targets = set(col_map.values())
        for col in batter_df.columns:
            if col in col_map:
                continue
            
            if '選手' in col and '選手名' not in mapped_targets:
                col_map[col] = '選手名'
                mapped_targets.add('選手名')
            elif 'woba' in col.lower() and 'wOBA' not in mapped_targets:
                col_map[col] = 'wOBA'
                mapped_targets.add('wOBA')
            elif '打席' in col and '打席' not in mapped_targets:
                col_map[col] = '打席'
                mapped_targets.add('打席')
            elif '試合' in col and '試合' not in mapped_targets:
                col_map[col] = '試合'
                mapped_targets.add('試合')
            elif ('区分名' in col or '対戦' in col or '球場' in col) and '区分名' not in mapped_targets:
                col_map[col] = '区分名'
                mapped_targets.add('区分名')
            elif '区分' in col and '区分' not in mapped_targets:
                col_map[col] = '区分'
                mapped_targets.add('区分')

        # リネーム実行と重複カラムの排除
        df_renamed = batter_df.rename(columns=col_map)
        df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]

        # 不足カラムを安全なデフォルト値で補完
        for col in ['選手名', '区分', '区分名', '試合', '打席', 'wOBA']:
            if col not in df_renamed.columns:
                if col in ['試合', '打席']:
                    df_renamed[col] = 0
                elif col == 'wOBA':
                    df_renamed[col] = league_avg
                elif col == '区分':
                    df_renamed[col] = ''
                else:
                    df_renamed[col] = ''

        # 必要な6カラムを確実に抽出
        df_woba = df_renamed[['選手名', '区分', '区分名', '試合', '打席', 'wOBA']].copy()

        # [Step 4-3] 安全＆超親切：もし「区分」列が空、または存在しなかった場合に備え、値から自動ラベリング
        team_names = set(target for _, target in TEAM_NAME_MAP)
        stadium_keywords = ['ドーム', '球場', 'スタジアム', 'フィールド', '宮城', 'マリン', '甲子園', '神宮', 'バンテリン', 'マツダ', 'PayPay', 'みずほ', 'エスコン', 'ほっともっと']

        def auto_classify_category(row):
            cat = str(row['区分']).strip()
            if cat in ['対戦相手', '球場']:
                return cat
            
            # 区分名から自動判別
            name = str(row['区分名']).strip()
            if name in team_names:
                return '対戦相手'
            if any(k in name for k in stadium_keywords):
                return '球場'
            
            # デフォルト
            return '対戦相手'

        df_woba['区分'] = df_woba.apply(auto_classify_category, axis=1)

        # [Step 4-4] ゴミデータを完全パージ
        invalid_values = {'対戦相手', '区分名', '選手名', '選手', '区分', '対戦', 'wOBA', 'woba'}
        df_woba = df_woba[
            ~df_woba['区分名'].astype(str).str.strip().isin(invalid_values) &
            ~df_woba['選手名'].astype(str).str.strip().isin(invalid_values)
        ]

    n_games = all_df['game_id'].nunique()
    n_pa = len(all_df)

    df_career = load_career_stats(stats_dir)
    df_team_runs = build_team_runs(df_runs)

    # 5. 球団別対戦成績
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

    # 6. 状況別成績
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
# 【予測ロジック：対戦相手・球場のダブル・ベイズ収縮補正対応】
# ============================================================
def predict_one(re24, df_woba, league_avg, batter, opponent, stadium, out, runner):
    """対戦相手補正と球場補正をダブルで考慮して、1つのシチュエーションの得点期待値を予測する"""
    try:
        val = re24.loc[out, runner]
        base_re = float(val) if pd.notna(val) else None
    except Exception:
        base_re = None

    # 打者全体の wOBA 算出
    df_all_b = df_woba[df_woba['選手名'] == batter]
    if not df_all_b.empty and df_all_b['打席'].sum() > 0:
        woba_overall = (df_all_b['wOBA'] * df_all_b['打席']).sum() / df_all_b['打席'].sum()
    else:
        woba_overall = league_avg

    C = 20  # 収縮定数

    # 1. 【対戦相手（球団）補正】
    opp_mult = 1.0
    opp_pa = 0
    opp_woba_pred = woba_overall
    if opponent and opponent != '全体':
        df_opp = df_woba[(df_woba['選手名'] == batter) & (df_woba['区分'] == '対戦相手') & (df_woba['区分名'] == opponent)]
        if not df_opp.empty:
            opp_pa = int(df_opp.iloc[0]['打席'])
            opp_woba_vs = float(df_opp.iloc[0]['wOBA'])
            opp_woba_pred = (opp_pa * opp_woba_vs + C * woba_overall) / (opp_pa + C)
        opp_mult = opp_woba_pred / league_avg if league_avg > 0 else 1.0

    # 2. 【球場補正】
    stad_mult = 1.0
    stad_pa = 0
    stad_woba_pred = woba_overall
    if stadium and stadium != '全体':
        df_stad = df_woba[(df_woba['選手名'] == batter) & (df_woba['区分'] == '球場') & (df_woba['区分名'] == stadium)]
        if not df_stad.empty:
            stad_pa = int(df_stad.iloc[0]['打席'])
            stad_woba_vs = float(df_stad.iloc[0]['wOBA'])
            stad_woba_pred = (stad_pa * stad_woba_vs + C * woba_overall) / (stad_pa + C)
        stad_mult = stad_woba_pred / league_avg if league_avg > 0 else 1.0

    # 最終的な期待得点の計算 (基礎期待値 × 対戦相手補正比率 × 球場補正比率)
    if base_re is not None:
        pred_re = base_re * opp_mult * stad_mult
    else:
        pred_re = None

    # 補足ノートの作成
    notes = []
    if opponent and opponent != '全体' and opp_pa < 10:
        notes.append(f"対 {opponent}（{opp_pa}打席）")
    if stadium and stadium != '全体' and stad_pa < 10:
        notes.append(f"球場 {stadium}（{stad_pa}打席）")
    
    note = None
    if notes:
        note = f"{'・'.join(notes)} のデータが少ないため、全体成績（wOBA: {woba_overall:.3f}）を加重して予測を補正しています。"

    # 総合的な補正後予測wOBA（対戦相手と球場の効果を掛け合わせたもの）
    combined_woba_pred = woba_overall * opp_mult * stad_mult

    return {
        '基礎RE24': round(base_re, 3) if base_re is not None else None,
        '打者wOBA': round(combined_woba_pred, 3),
        '補正後期待得点': round(pred_re, 3) if pred_re is not None else None,
        '対戦打席数': opp_pa,
        '球場打席数': stad_pa,
        'note': note
    }


def predict_all(re24, df_woba, league_avg, batter, opponent, stadium):
    """24場面すべての予測結果をDataFrameにまとめて返す"""
    rows = []
    for out in [0, 1, 2]:
        for runner in RUNNER_LABEL:
            res = predict_one(re24, df_woba, league_avg, batter, opponent, stadium, out, runner)
            rows.append({
                'アウト': out,
                'ランナー': runner,
                '打者wOBA': res['打者wOBA'],
                '基礎RE24': res['基礎RE24'],
                '補正後期待得点': res['補正後期待得点'],
                '対戦打席数': res['対戦打席数'],
                '球場打席数': res.get('球場打席数', 0)
            })
    return pd.DataFrame(rows)
