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
# scraper.py が _details.csv に追加した列（2026-11〜）。calc_runs_after で引き継ぐ
DETAIL_EXTRA_COLS = ['打者ID', '投手名', '投手ID', '攻撃']

# レギュラーシーズン以外の試合（スポナビの試合ID）。RE24 などはレギュラーシーズンだけで計算する
# （検証の設計: レギュラーシーズンで学習 → ポストシーズンでテスト。研究計画ドキュメント参照）
ALLSTAR_IDS = {2021039499, 2021039500}
POSTSEASON_IDS = {
    2021039474, 2021039475,                          # CS 1st セ（巨人 vs DeNA）
    2021039477, 2021039478, 2021039479,              # CS 1st パ（西武 vs 日本ハム）
    2021039480, 2021039481, 2021039482, 2021039483,  # CS ファイナル セ（阪神 vs DeNA）
    2021039486, 2021039487, 2021039488,              # CS ファイナル パ（ソフトバンク vs 日本ハム）
    2021039492, 2021039493, 2021039494, 2021039495,  # 日本シリーズ（DeNA vs ソフトバンク）
}
NON_REGULAR_IDS = ALLSTAR_IDS | POSTSEASON_IDS


def is_regular_season(game_id) -> bool:
    """レギュラーシーズンの試合か（オールスター・ポストシーズンでない）"""
    try:
        return int(game_id) not in NON_REGULAR_IDS
    except (TypeError, ValueError):
        return True


def find_last_score(body) -> re.Match | None:
    """
    本文の中の「最後の」得点表記（例: 広 3-5 ロ）を返す。
    1打席の本文に得点が2回書かれることがある（例: 「暴投 広 3-4 ロ … タイムリー 広 3-5 ロ」）。
    以前は最初の表記だけを読んでいたため、7試合でスコアボードと得点が合わなかった。
    """
    ms = list(RE_SCORE.finditer(str(body)))
    return ms[-1] if ms else None
# 対戦相手別 wOBA のベイズ収縮の強さ（＝事前分布の重みを「何打席分」とみなすか）
#   wOBA予測 = (PA×wOBA対戦 + C×wOBA全体) / (PA + C)
#   ポスター記載の値 C = 20。PA は wOBA の分母（打席 − 敬遠 − 犠打）
SHRINK_C = 20

# 球団の正式名称（カード表記）→ 状況別データの区分名に統一するマッピング
# ※ raw 側は normalize_team_name 内で NFKC 正規化してから照合するため、
#   key は半角/全角どちらの表記が来ても拾えるよう半角で統一しておく。
#   target（df_woba 側 = nf3 の区分名）は現地サイトの表記に合わせて
#   全角/半角どちらも吸収できるように候補を複数持たせる。
TEAM_NAME_MAP = [
    ('広島',     '広島'),
    ('中日',     '中日'),
    ('ロッテ',   'ロッテ'),
    ('西武',     '西武'),
    ('オリックス', 'オリックス'),
    ('楽天',     '楽天'),
    ('ソフトバンク', 'ソフトバンク'),
    ('日本ハム', '日本ハム'),
    ('読売',     '巨人'),
    ('巨人',     '巨人'),
    ('阪神',     '阪神'),
    ('DeNA',     'ＤｅＮＡ'),
    ('横浜',     'ＤｅＮＡ'),
    ('ヤクルト', 'ヤクルト'),
]

# nf3 側（df_woba の区分名）は表記が揺れることがあるため、実際に照合する際は
# 双方を NFKC 正規化してから比較する。ここでは normalize_team_name の出力を
# 「正規化済みの代表表記」に統一するための逆引き（正規化後 → 代表表記）も用意する。
_TEAM_CANONICAL = {
    unicodedata.normalize('NFKC', target): target
    for _key, target in TEAM_NAME_MAP
}


def normalize_name(s) -> str:
    """
    選手名・球団名などの表記ゆれを吸収するための共通正規化関数。
    - NFKC 正規化（全角英数字/記号 → 半角、互換文字の統一など）
    - 空白（半角・全角・タブ等）の除去
    - 中黒（・）の除去（外国人選手名の表記ゆれ対策）
    """
    if s is None:
        return ''
    s = unicodedata.normalize('NFKC', str(s))
    s = re.sub(r'[\s　]', '', s)
    s = s.replace('・', '')
    return s


def normalize_team_name(raw: str | None) -> str | None:
    """カードの長い球団名を状況別データの区分名（短縮名）に変換する"""
    if not raw:
        return None
    s = unicodedata.normalize('NFKC', raw)
    for key, target in TEAM_NAME_MAP:
        if unicodedata.normalize('NFKC', key) in s:
            # target を正規化した「代表表記」に統一して返す
            return _TEAM_CANONICAL.get(unicodedata.normalize('NFKC', target), target)
    return raw


# ============================================================
# 本文テキストから打席結果を判定するためのキーワードパターン
# 優先順位が重要：長打（二塁打/三塁打/本塁打）を単打より先に判定し、
# 「ツーベース」「スリーベース」等の表記ゆれにも対応する。
# けん制・投手交代・代打/代走・守備変更などの前置きノイズ文言には
# これらのキーワードが含まれないことをサンプルデータで確認済み。
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
    """
    本文テキストから打席結果コードを判定する。
    該当なしの場合は 'UNKNOWN' を返す（打率計算からは除外される）。
    """
    s = str(text) if text is not None else ''
    for code, pattern in RESULT_PATTERNS:
        if pattern.search(s):
            return code
    return 'UNKNOWN'


def _opponent_team_raw(row) -> str | None:
    """
    打者側から見た対戦（相手）球団の生の名前を返す。
    表イニング = アウェイチームが攻撃 → 対戦相手はホーム
    裏イニング = ホームチームが攻撃 → 対戦相手はアウェイ
    """
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
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['アウト']   = pd.to_numeric(all_df['アウト'],   errors='coerce').fillna(0).astype(int)
    all_df['打席順']  = pd.to_numeric(all_df['打席順'],  errors='coerce').fillna(0).astype(int)
    all_df['打順']    = pd.to_numeric(all_df['打順'],    errors='coerce')
    all_df['ランナー'] = all_df['ランナー'].fillna('')
    return all_df


# ============================================================
# Step 2: イニング残り得点を付与
# ============================================================
def _inning_sort_key(inning_str) -> tuple[int, int]:
    """
    'イニング' 列（例: '1回表', '5回裏'）を試合内の時系列順に並べるための
    ソートキーを返す。(イニング番号, 表=0/裏=1)
    """
    s = str(inning_str)
    m = re.search(r'(\d+)', s)
    num = int(m.group(1)) if m else 0
    tb = 1 if '裏' in s else 0
    return (num, tb)


def calc_runs_after(all_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for game_id, game_grp in all_df.groupby('game_id'):
        # 半イニングを試合内の時系列順（イニング番号→表/裏）に並べる。
        # 得点の累計（carry_score）をイニングをまたいで引き継ぐために、
        # 出現順ではなく明示的に時系列でソートする必要がある。
        inning_order = sorted(game_grp['イニング'].unique(), key=_inning_sort_key)

        carry_score = 0  # この試合でそれまでに記録された総得点（両チーム合計）
        for inning in inning_order:
            grp = game_grp[game_grp['イニング'] == inning].sort_values('打席順').reset_index(drop=True)

            score_after, last_score = [None] * len(grp), None
            for i, body in enumerate(grp['本文'].fillna('')):
                m = find_last_score(body)
                if m:
                    last_score = int(m.group(1)) + int(m.group(2))
                score_after[i] = last_score

            # このイニング開始前の得点（carry_score）を起点に増分（delta）を求める。
            # 以前は各イニングの起点を常に 0 としていたため、そのイニングの
            # 先頭打者（＝必ず 0アウト・走者なしの場面）に、それ以前の試合の
            # 総得点がまるごと加算されてしまい、RE24（特に 0アウト・走者なし）
            # の値が異常に高くなる不具合があった。
            delta, prev = [], carry_score
            for s in score_after:
                cur = s if s is not None else prev
                delta.append(cur - prev)
                prev = cur

            suffix = np.cumsum(delta[::-1])[::-1]

            # 次の半イニングの起点用に、このイニングの最終得点を引き継ぐ
            carry_score = prev

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
                    # 打者ID・投手ID など（scraper.EXTRA_COLS）。旧形式の CSV では None
                    **{c: row.get(c) for c in DETAIL_EXTRA_COLS},
                })
    return pd.DataFrame(results)


# ============================================================
# 選手マスタ（data/players.csv）の結合
# ============================================================
def normalize_player_id(v) -> str | None:
    """
    選手IDを文字列に揃える。CSV 読み込み時に欠損があると 1600123.0 のような
    float になるため、そのままでは players.csv の ID と一致しない。
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s or None


def attach_player_info(df_runs: pd.DataFrame, df_players: pd.DataFrame | None) -> pd.DataFrame:
    """
    打席データ（calc_runs_after の出力）に、選手マスタから打者・投手の属性を付ける。

    追加する列:
        打者_打        : 打者の登録上の打席（右/左/両）
        打者_打席      : この打席で実際に立った側（右/左）。両打は投手の逆側とみなす
        投手_腕        : 投手の投げ腕（右/左）
        投手_直球球速  : 投手のストレート平均球速（km/h）
        投手_平均球速  : 投手の全球種の加重平均球速（km/h）
        同側対戦       : 打者_打席 == 投手_腕（右対右・左対左）なら True

    打者ID・投手ID の列がない旧形式のデータや、マスタに無い選手は NaN のまま。
    """
    df = df_runs.copy()
    for c in ('打者ID', '投手ID'):
        if c not in df.columns:
            df[c] = None
        df[c] = df[c].apply(normalize_player_id)

    if df_players is None or df_players.empty:
        for c in ('打者_打', '打者_打席', '投手_腕', '投手_直球球速', '投手_平均球速', '同側対戦'):
            df[c] = np.nan
        return df

    pl = df_players.copy()
    pl['選手ID'] = pl['選手ID'].apply(normalize_player_id)
    pl = pl.drop_duplicates('選手ID').set_index('選手ID')

    df['打者_打']       = df['打者ID'].map(pl['打'])
    df['投手_腕']       = df['投手ID'].map(pl['投'])
    df['投手_直球球速'] = df['投手ID'].map(pl['直球_平均球速'])
    df['投手_平均球速'] = df['投手ID'].map(pl['全球種_平均球速'])

    opposite = {'右': '左', '左': '右'}
    df['打者_打席'] = np.where(df['打者_打'] == '両',
                               df['投手_腕'].map(opposite),
                               df['打者_打'])
    df['打者_打席'] = df['打者_打席'].where(df['打者_打席'].isin(['右', '左']))
    known = df['打者_打席'].notna() & df['投手_腕'].notna()
    df['同側対戦'] = (df['打者_打席'] == df['投手_腕']).where(known)
    return df


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
# Step 3.6: 対戦球団ごとの打率・得点期待値（状況別ではない）
# ============================================================
def build_team_batting_stats(df_runs: pd.DataFrame, min_pa: int = 1) -> pd.DataFrame:
    """
    実際のプレーバイプレー（本文）から、選手×対戦球団ごとの
    打率と得点期待値（状況を問わず runs_after の平均）を集計する。

    打率 = 安打 / 打数（打数 = 打席 - 四球 - 死球 - 敬遠 - 犠打 - 犠飛）
    得点期待値 = その選手がその球団と対戦した全打席の runs_after の平均
                （アウト・ランナー状況では区切らない）

    min_pa 未満の標本は打率・得点期待値を None にする（信頼性が低いため）。
    デフォルトは1打席以上あれば数値を表示する（交流戦のように打席数が
    少ない対戦相手でも実測値が見えるようにするため）。信頼性の判断は
    打席数の列を見て利用側で行うことを想定している。

    Returns
    -------
    DataFrame [選手名, 対戦球団, 打席, 打数, 安打, 打率, 得点期待値]
    """
    df = add_result_columns(df_runs)
    df = df[df['対戦球団'].notna()]

    # _details.csv の選手名は「田中 幹也」のように半角/全角スペース入りで
    # 記録されているが、対戦成績CSV（batter_csv）側は「田中幹也」とスペースなし。
    # また外国人選手名は中黒（・）や全角/半角の表記ゆれもある。
    # df_woba 側の '選手名_key'（normalize_name 適用済み）と完全に同じ正規化
    # ルールを適用しないと突き合わせが一致せず、「対戦データ（nf3）はあるのに
    # 実測データが出ない」という不具合につながるため、ここも normalize_name で揃える。
    df['選手名'] = df['選手名'].astype(str).apply(normalize_name)

    rows = []
    for (batter, team), grp in df.groupby(['選手名', '対戦球団']):
        pa   = len(grp)
        ab   = int(grp['is_ab'].sum())
        hits = int(grp['is_hit'].sum())
        avg  = (hits / ab) if ab > 0 else np.nan
        exp_runs = float(grp['runs_after'].mean())

        enough = pa >= min_pa
        rows.append({
            '選手名':     batter,
            '対戦球団':   team,
            '打席':       pa,
            '打数':       ab,
            '安打':       hits,
            '打率':       round(avg, 3) if (enough and not np.isnan(avg)) else None,
            '得点期待値': round(exp_runs, 3) if enough else None,
        })

    return pd.DataFrame(rows).sort_values(['選手名', '対戦球団']).reset_index(drop=True)


# ============================================================
# Step 3.7: 状況別（アウト×ランナー）の打率・得点期待値
# ============================================================
def build_situational_stats(df_runs: pd.DataFrame, min_pa: int = 5) -> pd.DataFrame:
    """
    実際のプレーバイプレーから、アウト×ランナー状況ごとの
    打率と得点期待値を集計する（対戦球団・選手は問わず全体）。
    既存の build_re24（得点期待値のみ）を打率つきに拡張したもの。

    Returns
    -------
    DataFrame [アウト, ランナー状態, 打席, 打数, 安打, 打率, 得点期待値]
    """
    df = add_result_columns(df_runs)
    df['runner_idx'] = df['ランナー'].map(RUNNER_MAP).fillna(0).astype(int)

    rows = []
    for out in range(3):
        for r_idx, r_label in enumerate(RUNNER_LABEL):
            grp = df[(df['アウト'] == out) & (df['runner_idx'] == r_idx)]
            pa  = len(grp)

            if pa == 0:
                rows.append({
                    'アウト': out, 'ランナー状態': r_label,
                    '打席': 0, '打数': 0, '安打': 0,
                    '打率': None, '得点期待値': None,
                })
                continue

            ab   = int(grp['is_ab'].sum())
            hits = int(grp['is_hit'].sum())
            avg  = (hits / ab) if ab > 0 else np.nan
            exp_runs = float(grp['runs_after'].mean())

            enough = pa >= min_pa
            rows.append({
                'アウト':       out,
                'ランナー状態': r_label,
                '打席':         pa,
                '打数':         ab,
                '安打':         hits,
                '打率':         round(avg, 3) if (enough and not np.isnan(avg)) else None,
                '得点期待値':   round(exp_runs, 3) if enough else None,
            })

    return pd.DataFrame(rows)


# ============================================================
# Step 4: 打者 wOBA テーブルを構築
# ============================================================
def build_batter_woba(df_raw: pd.DataFrame):
    df = df_raw[df_raw['区分種別'] == '対戦相手'].copy()
    df['選手名_key'] = df['選手名'].apply(normalize_name)
    # nf3 側の区分名（対戦相手の球団表記）を、_details.csv 側（プレーバイプレー）
    # の対戦球団表記と同じ「代表表記」に統一する。表記が食い違ったままだと
    # （例: 半角 'DeNA' と全角 'ＤｅＮＡ'）、実測打率・得点期待値のルックアップが
    # 常に空になり「対戦データはあるのに実測データが出ない」という状態になる。
    df['区分名'] = df['区分名'].apply(normalize_team_name)
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
    # wOBA_num / wOBA_den はベイズ収縮（predict_batter_woba）で使う
    return (df[['選手名', '選手名_key', '区分名', '球団', '試合', '打席', 'wOBA', 'wOBA_num', 'wOBA_den']].copy(),
            float(league_avg))


# ============================================================
# Step 4.2: 対戦相手別 wOBA のベイズ収縮
# ============================================================
def predict_batter_woba(df_woba: pd.DataFrame, batter_name: str, opponent: str,
                        league_avg: float, c: float = SHRINK_C) -> dict:
    """
    打者の「対戦相手別 wOBA」を、その打者の全体 wOBA に向けて収縮させた予測値を返す。

        wOBA予測 = (PA × wOBA対戦 + C × wOBA全体) / (PA + C)
                 = (wOBA分子_対戦 + C × wOBA全体) / (wOBA分母_対戦 + C)

      PA        : 対戦相手との wOBA の分母（打席 − 敬遠 − 犠打）
      wOBA全体  : 全対戦相手を合計した wOBA（分子の合計 / 分母の合計）
      C         : 収縮の強さ（SHRINK_C = 20 打席分）

    対戦打席が少ないほど全体 wOBA に近づき、多いほど対戦 wOBA に近づく。
    以前の「分母 10 未満なら全体平均に切り替え」という不連続な扱いを置き換える。
    選手データ自体がない場合はリーグ平均 wOBA を返す。

    Returns
    -------
    dict: wOBA予測, wOBA対戦（生の値・分母0ならNone）, wOBA全体, 対戦PA, 対戦試合数, 収縮率（C/(PA+C)）, note
    """
    key = normalize_name(batter_name)
    sub = df_woba[df_woba['選手名_key'] == key].copy()
    if 'wOBA_num' not in sub.columns or 'wOBA_den' not in sub.columns:
        # 古い build_batter_woba で作った表（wOBA と打席だけ）への互換: 打席数を分母とみなす
        sub['wOBA_den'] = sub['打席'].where(sub['wOBA'].notna(), 0)
        sub['wOBA_num'] = (sub['wOBA'] * sub['打席']).fillna(0)
    num_all = float(sub['wOBA_num'].sum()) if not sub.empty else 0.0
    den_all = float(sub['wOBA_den'].clip(lower=0).sum()) if not sub.empty else 0.0

    if den_all <= 0:
        return {'wOBA予測': league_avg, 'wOBA対戦': None, 'wOBA全体': None,
                '対戦PA': 0, '対戦試合数': 0, '収縮率': 1.0,
                'note': f'※ {batter_name} の打撃データなし → リーグ平均wOBAを使用'}

    woba_all = num_all / den_all
    hit = sub[sub['区分名'] == opponent]
    num_o = float(hit['wOBA_num'].sum()) if not hit.empty else 0.0
    den_o = float(hit['wOBA_den'].clip(lower=0).sum()) if not hit.empty else 0.0
    games = int(hit['試合'].sum()) if not hit.empty else 0

    pred   = (num_o + c * woba_all) / (den_o + c)
    shrink = c / (den_o + c)
    note = ''
    if den_o <= 0:
        note = f'※ {batter_name} vs {opponent} の対戦データなし → {batter_name} の全体wOBAを使用'
    return {'wOBA予測': pred,
            'wOBA対戦': (num_o / den_o) if den_o > 0 else None,
            'wOBA全体': woba_all,
            '対戦PA': int(den_o), '対戦試合数': games,
            '収縮率': shrink, 'note': note}


# ============================================================
# Step 4.5: 選手の全対戦相手平均 wOBA
#   （対戦相手別データがない場合のフォールバック用）
# ============================================================
def get_batter_avg_woba(df_woba: pd.DataFrame, batter_name: str) -> tuple[float | None, int, int]:
    """
    指定選手の「全対戦相手」を通した wOBA を、打席数で加重平均して返す。
    対戦相手ごとのデータが1件もない、または全て打席不足で wOBA が NaN の場合は
    (None, 0, 0) を返す（呼び出し側でリーグ平均にフォールバックする）。

    Returns
    -------
    (avg_woba, total_pa, total_games)
    """
    key = normalize_name(batter_name)
    sub = df_woba[df_woba['選手名_key'] == key]
    if sub.empty:
        return None, 0, 0

    valid = sub[sub['wOBA'].notna() & (sub['打席'] > 0)]
    if valid.empty:
        return None, 0, 0

    total_pa      = int(valid['打席'].sum())
    weighted_woba = float((valid['wOBA'] * valid['打席']).sum() / total_pa)
    total_games   = int(valid['試合'].sum())
    return weighted_woba, total_pa, total_games


# ============================================================
# Step 4.6: 今シーズンの打者成績（通算）
#   対戦相手別の成績（nf3）を、選手×球団ごとに合計してシーズン通算にする。
#   1試合は必ず1つの対戦相手との試合なので、試合数や打席数を合計して構わない。
# ============================================================
SEASON_SUM_COLS = ['試合', '打席', '打数', '得点', '安打', '2B', '3B', '本塁', '塁打', '打点',
                   '盗塁', '盗刺', '犠打', '犠飛', '四球', '敬遠', '死球', '三振', '併殺']


def build_season_batting(batter_df: pd.DataFrame) -> pd.DataFrame:
    """
    all_batters_situational.csv 相当の DataFrame から、選手×球団ごとの
    シーズン成績（打率・出塁率・長打率・OPS・wOBA など）を作る。

    CSVに無い列は飛ばし、打数・塁打が無ければ打席などから補う。
    wOBA は build_batter_woba と同じ式（四球をそのまま加算、分母は打席 - 敬遠 - 犠打）。
    必要な列（打席・安打）が無い場合は空の DataFrame を返す。
    """
    if batter_df is None or batter_df.empty or '区分種別' not in batter_df.columns:
        return pd.DataFrame()

    df = batter_df[batter_df['区分種別'] == '対戦相手'].copy()
    if df.empty or not {'選手名', '打席', '安打'}.issubset(df.columns):
        return pd.DataFrame()
    if '球団' not in df.columns:
        df['球団'] = None

    sum_cols = [c for c in SEASON_SUM_COLS if c in df.columns]
    for c in sum_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    g = (df.groupby(['選手名', '球団'], dropna=False)[sum_cols]
           .sum(min_count=1).reset_index())
    g['選手名_key'] = g['選手名'].apply(normalize_name)

    def col(name):
        return g[name].fillna(0) if name in g.columns else 0

    pa, h = g['打席'].fillna(0), g['安打'].fillna(0)
    b2, b3, hr = col('2B'), col('3B'), col('本塁')
    bb, hbp, sh, sf, ibb = col('四球'), col('死球'), col('犠打'), col('犠飛'), col('敬遠')

    ab = g['打数'] if '打数' in g.columns else (pa - bb - hbp - sh - sf)
    tb = g['塁打'] if '塁打' in g.columns else (h + b2 + 2 * b3 + 3 * hr)
    g['打数'], g['塁打'] = ab, tb

    obp_den = ab + bb + hbp + sf
    with np.errstate(divide='ignore', invalid='ignore'):
        g['打率']   = np.where(ab > 0, h / ab, np.nan)
        g['出塁率'] = np.where(obp_den > 0, (h + bb + hbp) / obp_den, np.nan)
        g['長打率'] = np.where(ab > 0, tb / ab, np.nan)
        g['OPS']    = g['出塁率'] + g['長打率']

        singles = h - b2 - b3 - hr
        num = (WOBA_W['BB'] * bb + WOBA_W['HBP'] * hbp + WOBA_W['S'] * singles +
               WOBA_W['D'] * b2 + WOBA_W['T'] * b3 + WOBA_W['HR'] * hr)
        den = pa - ibb - sh
        g['wOBA'] = np.where(den > 0, num / den, np.nan)

    order = ['選手名', '選手名_key', '球団', '試合', '打席', '打数', '得点', '安打', '2B', '3B', '本塁',
             '塁打', '打点', '盗塁', '四球', '死球', '三振', '犠打', '犠飛',
             '打率', '出塁率', '長打率', 'OPS', 'wOBA']
    return (g[[c for c in order if c in g.columns]]
            .sort_values('打席', ascending=False).reset_index(drop=True))


# ============================================================
# Step 5: 1場面の予測
# ============================================================
def predict_one(re24, df_woba, league_avg,
                batter_name: str, opponent: str,
                out: int, runner: str) -> dict:
    runner_idx = RUNNER_MAP.get(runner, 0)
    base_re    = re24.iloc[out, runner_idx]

    # 対戦相手別 wOBA を全体 wOBA へベイズ収縮した値を打者の能力とする（ポスターの式）
    est         = predict_batter_woba(df_woba, batter_name, opponent, league_avg)
    batter_woba = est['wOBA予測']
    pa          = est['対戦PA']
    games       = est['対戦試合数']
    note        = est['note']

    adj_re = (base_re * (batter_woba / league_avg)
              if not np.isnan(base_re) and league_avg > 0 else np.nan)

    return {
        'アウト':         out,
        'ランナー':       runner if runner else '走者なし',
        '打者wOBA':       round(batter_woba, 3),
        '対戦wOBA(生)':   round(est['wOBA対戦'], 3) if est['wOBA対戦'] is not None else None,
        '全体wOBA':       round(est['wOBA全体'], 3) if est['wOBA全体'] is not None else None,
        '収縮率':         round(est['収縮率'], 3),
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
        except Exception:
            continue

        # ファイル名（例: stats_2024.csv）から年度を推定しておく。
        # CSV内の '年度' 列が欠落している、または一部の行だけ空になっている
        # ケースがあり、それをそのまま結合すると該当年の '年度' が NaN のまま
        # （画面上は None）表示されてしまうため、ファイル名側の年度で補完する。
        m = re.search(r'(\d{4})', os.path.basename(f))
        file_year = int(m.group(1)) if m else None

        if '年度' not in df.columns:
            df['年度'] = file_year
        else:
            df['年度'] = pd.to_numeric(df['年度'], errors='coerce')
            if file_year is not None:
                df['年度'] = df['年度'].fillna(file_year)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    all_stats = pd.concat(dfs, ignore_index=True)
    all_stats['年度'] = all_stats['年度'].astype('Int64')
    # stats_2023.csv に完全に同じ行が重複している選手がいる（リチャード、秋広など4人）
    all_stats = all_stats.drop_duplicates(subset=['選手名', '所属球団', '年度', '試合', '打席']).reset_index(drop=True)
    all_stats['選手名_key'] = all_stats['選手名'].apply(normalize_name)
    all_stats['OPS'] = all_stats['出塁率'] + all_stats['長打率']
    return all_stats


def get_player_career(df_stats: pd.DataFrame, batter_name: str) -> pd.DataFrame:
    """
    指定選手の過去成績を年度昇順で返す。
    """
    if df_stats.empty:
        return pd.DataFrame()
    key = normalize_name(batter_name)
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
def build_model_from_dfs(dfs: list[pd.DataFrame], batter_df: pd.DataFrame, stats_dir: str | None = None):
    """
    dfs: GitHub から読み込んだ _details.csv の DataFrame リスト
    batter_df: 対戦相手別・球場別打撃成績（all_batters_situational.csv 相当）の DataFrame
    stats_dir: 過去3年成績 CSV が入っているディレクトリ（任意）
    """
    # オールスター・ポストシーズンは RE24 などの計算から除く
    dfs = [d for d in dfs if 'game_id' not in d.columns or len(d) == 0 or is_regular_season(d['game_id'].iloc[0])]
    if dfs:
        all_df   = concat_details(dfs)
        df_runs  = calc_runs_after(all_df)
        re24, counts = build_re24(df_runs)
        df_team_runs = build_team_game_runs(df_runs)
        df_team_batting_stats  = build_team_batting_stats(df_runs)
        df_situational_stats   = build_situational_stats(df_runs)
        n_pa     = len(all_df)
    else:
        re24 = pd.DataFrame(
            np.full((3, 8), np.nan),
            index=pd.Index([0, 1, 2], name='アウト'),
            columns=pd.Index(RUNNER_LABEL, name='ランナー状態'),
        )
        counts       = np.zeros((3, 8), dtype=int)
        df_team_runs = pd.DataFrame(columns=['game_id', 'team', 'runs'])
        df_team_batting_stats = pd.DataFrame(
            columns=['選手名', '対戦球団', '打席', '打数', '安打', '打率', '得点期待値'])
        df_situational_stats = pd.DataFrame(
            columns=['アウト', 'ランナー状態', '打席', '打数', '安打', '打率', '得点期待値'])
        n_pa         = 0

    df_woba, league_avg = build_batter_woba(batter_df)
    df_career = load_career_stats(stats_dir) if stats_dir else pd.DataFrame()

    return (re24, counts, df_woba, league_avg, len(dfs), n_pa, df_career, df_team_runs,
            df_team_batting_stats, df_situational_stats)
