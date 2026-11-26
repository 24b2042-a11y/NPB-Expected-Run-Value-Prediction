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
# wOBA の係数（BB=敬遠を除く四球、HBP=死球、S=単打、D=二塁打、T=三塁打、HR=本塁打）
#   MLB: 以前使っていた MLB の係数（The Book / FanGraphs の代表値）
#   NPB2026: 2026年 NPB レギュラーシーズンの一球速報から算出（core.compute_woba_weights、2026-11-26）
#     各結果の得点価値 = その結果の打席の ΔRE（RE24 の変化 + 入った得点）の平均
#     係数 = (結果の得点価値 − 凡退の得点価値 −0.231) × wOBA スケール 1.344
#     wOBA スケール = リーグ出塁率 0.306 ÷（スケール前のリーグ wOBA）… リーグ平均 wOBA が出塁率と同じになるように調整
WOBA_W_MLB     = dict(BB=0.70, HBP=0.73, S=0.89, D=1.27, T=1.61, HR=2.10)
WOBA_W_NPB2026 = dict(BB=0.726, HBP=0.735, S=0.889, D=1.360, T=1.858, HR=2.164)
WOBA_W         = WOBA_W_NPB2026   # アプリ・検証で使う係数
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
    if not isinstance(raw, str) or not raw:
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
    ('HBP', re.compile(r'死球|デッドボール')),
    ('SO',  re.compile(r'三振|振り逃げ')),
    ('1B',  re.compile(r'ヒット|安打')),
    ('SF',  re.compile(r'犠飛|犠牲フライ')),
    ('SAC', re.compile(r'バント')),
    # 併殺は「ダブルプレー」、失策出塁は「悪送球」「ファンブル」「後逸」などと書かれる（2026-11-26 追加）。
    # 失策での出塁は wOBA の慣例どおり凡退として数える
    ('OUT', re.compile(r'ゴロ|フライ|邪飛|ライナー|失策|エラー|併殺|ダブルプレー|トリプルプレー|悪送球|ファンブル|後逸|落球|野選|フィルダースチョイス|打撃妨害|守備妨害')),
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
    # 裏 = ホームの攻撃なので対戦相手はアウェイ（_opponent_team_raw と同じ判定を列で計算）
    is_bottom = df['イニング'].astype(str).str.contains('裏')
    df['対戦球団_raw'] = df['away_team_raw'].where(is_bottom, df['home_team_raw'])
    team_map = {t: normalize_team_name(t) for t in df['対戦球団_raw'].dropna().unique()}
    df['対戦球団']     = df['対戦球団_raw'].map(team_map)
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
    """
    各打席に「その打席からイニング終了までに入った得点（runs_after）」を付ける。

    本文の得点表記（例: 広 3-5 ロ）から両チーム合計の得点を読み、試合の中で時系列（イニング番号→表/裏→打席順）に
    並べて前の打席との差（その打席で入った得点）を出し、半イニングの中で後ろから足し上げる。
    得点表記のない打席は直前の得点のまま（差 0）。各半イニングの起点は、それまでの試合の得点。
    （2026-11-26 に行ごとのループをやめて列の計算に書き直した。結果は以前の実装と同じ）
    """
    if all_df.empty:
        return pd.DataFrame(columns=['game_id', 'イニング', '打席順', '打順', '選手名', 'アウト', 'ランナー', '本文',
                                     'runs_after', 'home_team_raw', 'away_team_raw', *DETAIL_EXTRA_COLS])
    df = all_df.copy()
    for c in ('打順', 'home_team_raw', 'away_team_raw', *DETAIL_EXTRA_COLS):
        if c not in df.columns:
            df[c] = None
    key = df['イニング'].map(_inning_sort_key)
    df['_inn_no'] = key.map(lambda k: k[0])
    df['_inn_tb'] = key.map(lambda k: k[1])
    df = df.sort_values(['game_id', '_inn_no', '_inn_tb', '打席順'], kind='stable').reset_index(drop=True)

    # 本文の最後の得点表記 → 両チーム合計。なければ NaN にして、試合の中で直前の値（最初は 0）で埋める
    def _total(body):
        m = find_last_score(body)
        return int(m.group(1)) + int(m.group(2)) if m else np.nan
    score = df['本文'].fillna('').map(_total)
    cur = score.groupby(df['game_id']).ffill().fillna(0)
    prev = cur.groupby(df['game_id']).shift(1).fillna(0)
    delta = cur - prev

    # 半イニングの中で後ろから累積和
    half = [df['game_id'], df['_inn_no'], df['_inn_tb']]
    df['runs_after'] = delta[::-1].groupby([h[::-1] for h in half]).cumsum()[::-1].astype(int)

    # 半イニング内の home/away は先頭行の値（従来どおり）
    for c in ('home_team_raw', 'away_team_raw'):
        df[c] = df.groupby(half)[c].transform('first')
    cols = ['game_id', 'イニング', '打席順', '打順', '選手名', 'アウト', 'ランナー', '本文',
            'runs_after', 'home_team_raw', 'away_team_raw', *DETAIL_EXTRA_COLS]
    return df[cols].reset_index(drop=True)


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
    # nf3 の「四球」は敬遠を含む（打席 = 打数+四球+死球+犠打+犠飛 が成り立つ）ので、敬遠を引いてから掛ける
    df['wOBA_num']  = (WOBA_W['BB']  * (df['四球'] - df['敬遠']).clip(lower=0) +
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
        num = (WOBA_W['BB'] * (bb - ibb).clip(lower=0) + WOBA_W['HBP'] * hbp + WOBA_W['S'] * singles +
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
def build_model_from_dfs(dfs: list[pd.DataFrame], batter_df: pd.DataFrame, stats_dir: str | None = None,
                         df_players: pd.DataFrame | None = None):
    """
    dfs: GitHub から読み込んだ _details.csv の DataFrame リスト
    batter_df: 対戦相手別・球場別打撃成績（all_batters_situational.csv 相当）の DataFrame
    stats_dir: 過去3年成績 CSV が入っているディレクトリ（任意）
    """
    # 調子の判定はポストシーズンも含めた直近の試合で行う（オールスターだけ除く）
    dfs_recent = [d for d in dfs if 'game_id' in d.columns and len(d)
                  and int(d['game_id'].iloc[0]) not in ALLSTAR_IDS]
    # オールスター・ポストシーズンは RE24 などの計算から除く
    dfs = [d for d in dfs if 'game_id' not in d.columns or len(d) == 0 or is_regular_season(d['game_id'].iloc[0])]
    if dfs:
        all_df   = concat_details(dfs)
        # calc_runs_after は時間がかかるので、ポストシーズンも含めて1回だけ計算し、レギュラーシーズン分を取り出す
        runs_recent = calc_runs_after(concat_details(dfs_recent))
        df_runs  = runs_recent[runs_recent['game_id'].map(is_regular_season)].reset_index(drop=True)
        re24, counts = build_re24(df_runs)
        df_team_runs = build_team_game_runs(df_runs)
        df_team_batting_stats  = build_team_batting_stats(df_runs)
        df_situational_stats   = build_situational_stats(df_runs)
        n_pa     = len(all_df)
        # 打者タイプ × 投手タイプの相性モデル（打者ID・投手ID の列がある新形式のデータのときだけ）
        type_model = (build_type_model(df_runs, df_players, re24)
                      if {'打者ID', '投手ID'}.issubset(df_runs.columns) and df_runs['投手ID'].notna().any()
                      else None)
        if type_model is not None:
            # 調子の判定表と、試合シミュレーション用のモデル（game_sim.py）も一緒に作っておく
            from game_sim import build_sim_model   # game_sim は core を import するので、ここで読み込む
            type_model['form'] = build_form_table(runs_recent, type_model)
            type_model['sim'] = build_sim_model(df_runs, df_players, type_model)
            # wOBA の係数の算出結果（表示用）と、条件別 wOBA 用の打席の表
            type_model['woba_weights'] = compute_woba_weights(df_runs, re24)
            type_model['pa'] = build_pa_table(runs_recent, df_players, type_model)
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
        type_model   = None

    df_woba, league_avg = build_batter_woba(batter_df)
    df_career = load_career_stats(stats_dir) if stats_dir else pd.DataFrame()

    return (re24, counts, df_woba, league_avg, len(dfs), n_pa, df_career, df_team_runs,
            df_team_batting_stats, df_situational_stats, type_model)


# ============================================================
# Step 8: 打者タイプ × 投手タイプの相性モデル（2026-11-26）
#
#   予測式（足し算の形。検証で乗法型より良かった）
#     補正後期待得点 = RE24[アウト, 走者] + k × (w − リーグ平均wOBA)
#     w = 打者wOBA + (投手の被wOBA − リーグ平均) + 相性[打者タイプ, 投手タイプ]
#   k: 学習データで「打席の得点価値 ΔRE」を (打者wOBA − リーグ平均) に回帰した係数（約0.78）
#
#   タイプ（2 × 2 = 4 通りずつ。相性は 4 × 4 = 16 マス）
#     打者: この打席の打席（右/左。両打は投手の逆側）× 長打型/巧打型（ISO が中央値以上なら長打型）
#     投手: 投げ腕（右/左）× 速球派/技巧派（速球の平均球速と奪三振率 K% の z 値の平均が 0 以上なら速球派）
#
#   どれも一球速報（レギュラーシーズン）から計算し、打席の少ない選手・マスはベイズ収縮する
#     打者wOBA = (分子 + C_BAT × lg) / (分母 + C_BAT)
#     投手の被wOBA = (分子 + C_PIT × lg) / (分母 + C_PIT)
#     相性 = Σ(実際の wOBA 分子 − 期待値) / (Σ分母 + C_MATCH)   期待値 = 打者wOBA + 投手の被wOBA − lg
# ============================================================
C_BAT   = 100   # 打者 wOBA をリーグ平均へ寄せる強さ（打席数）。検証で最良だった値
C_PIT   = 100   # 投手の被 wOBA をリーグ平均へ寄せる強さ
C_MATCH = 300   # タイプ相性を 0 へ寄せる強さ
C_ISO   = 100   # タイプ分類用の ISO をリーグ平均へ寄せる強さ（打数）
C_K     = 70    # タイプ分類用の K% をリーグ平均へ寄せる強さ（対戦打者数）
MIN_PA_FOR_THRESHOLD = 100   # タイプの境目（中央値）を決めるときに使う最低打席数・対戦打者数

# 一球速報の打席結果コード → wOBA の分子の重み
WOBA_VALUE = {'BB': WOBA_W['BB'], 'HBP': WOBA_W['HBP'], '1B': WOBA_W['S'],
              '2B': WOBA_W['D'], '3B': WOBA_W['T'], 'HR': WOBA_W['HR']}
FASTBALLS = ('ストレート', 'ツーシーム', 'シュート')


def add_woba_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    打席ごとの wOBA の分子・分母、ISO・K% 用の列を付ける（add_result_columns の後に使う）。
      woba_num: 四球0.70・死球0.73・単打0.89・二塁打1.27・三塁打1.61・本塁打2.10（敬遠・凡退は 0）
      woba_den: 1（敬遠・犠打と、結果を判定できなかった打席は 0）
    """
    out = df.copy()
    known = out['結果'] != 'UNKNOWN'
    out['woba_num'] = out['結果'].map(WOBA_VALUE).fillna(0.0).where(known, 0.0)
    out['woba_den'] = (known & ~out['結果'].isin(['IBB', 'SAC'])).astype(int)
    out['tb_extra'] = out['結果'].map({'2B': 1, '3B': 2, 'HR': 3}).fillna(0)   # ISO の分子（長打の余分な塁打）
    out['is_so'] = (out['結果'] == 'SO').astype(int)
    out['is_bf'] = known.astype(int)
    return out


def fastball_velocity(pitch_data) -> float | None:
    """
    players.csv の 球種データ（「球種:最高:平均:割合;…」）から速球の平均球速を返す。
    ストレートがあればその平均、なければツーシーム・シュートのうち速い方。
    （ストレートを投げない投手が「技巧派」に入ってしまうのを防ぐ）
    """
    if not isinstance(pitch_data, str) or not pitch_data:
        return None
    velo = {}
    for item in pitch_data.split(';'):
        parts = item.split(':')
        if len(parts) >= 3 and parts[0] in FASTBALLS:
            try:
                velo[parts[0]] = float(parts[2])
            except ValueError:
                pass
    if 'ストレート' in velo:
        return velo['ストレート']
    return max(velo.values()) if velo else None


def _zscore(s: pd.Series, ref: pd.Series) -> pd.Series:
    sd = ref.std()
    return (s - ref.mean()) / sd if sd and sd > 0 else s * 0


def build_type_model(df_runs: pd.DataFrame, df_players: pd.DataFrame | None, re24: pd.DataFrame,
                     c_bat: float = C_BAT, c_pit: float = C_PIT, c_match: float = C_MATCH,
                     use_style: bool = True) -> dict:
    """
    打席データ（calc_runs_after の出力。レギュラーシーズンのみを渡す）から相性モデルを作る。

    Returns: dict
        lg       : リーグ平均 wOBA（打席加重）
        k        : 足し算の式の係数
        batters  : DataFrame（index=打者ID）打席, wOBA, w（収縮後）, ISO, 長打型, 打, 選手名
        pitchers : DataFrame（index=投手ID）対戦打者, 被wOBA, w（収縮後）, K%, 速球, 速球派, 投, 選手名, 球団
        matchup  : DataFrame（index=打者タイプ, columns=投手タイプ）相性（wOBA の差）
        matchup_n: 同じ形の打席数
        params   : 収縮の強さなど
    use_style=False にすると長打型/巧打型・速球派/技巧派を使わず、左右だけの相性になる（検証の比較用）
    """
    df = add_woba_columns(add_result_columns(df_runs))
    df = attach_player_info(df, df_players)
    df = df[df['打者ID'].notna() & df['投手ID'].notna()]

    lg = float(df['woba_num'].sum() / df['woba_den'].sum())

    # --- 打者 ---
    b = df.groupby('打者ID').agg(打席=('woba_den', 'sum'), num=('woba_num', 'sum'),
                                ab=('is_bf', 'sum'), tb_extra=('tb_extra', 'sum'),
                                選手名=('選手名', 'first'))
    b['wOBA'] = (b['num'] / b['打席']).where(b['打席'] > 0)
    b['w'] = (b['num'] + c_bat * lg) / (b['打席'] + c_bat)
    lg_iso = b['tb_extra'].sum() / b['ab'].sum()
    b['ISO'] = (b['tb_extra'] + C_ISO * lg_iso) / (b['ab'] + C_ISO)
    iso_cut = b.loc[b['打席'] >= MIN_PA_FOR_THRESHOLD, 'ISO'].median()
    b['長打型'] = b['ISO'] >= iso_cut

    # --- 投手 ---
    p = df.groupby('投手ID').agg(対戦打者=('woba_den', 'sum'), num=('woba_num', 'sum'),
                                bf=('is_bf', 'sum'), so=('is_so', 'sum'), 選手名=('投手名', 'first'))
    p['被wOBA'] = (p['num'] / p['対戦打者']).where(p['対戦打者'] > 0)
    p['w'] = (p['num'] + c_pit * lg) / (p['対戦打者'] + c_pit)
    lg_k = p['so'].sum() / p['bf'].sum()
    p['K%'] = (p['so'] + C_K * lg_k) / (p['bf'] + C_K)
    if df_players is not None and not df_players.empty:
        pl = df_players.copy()
        pl['選手ID'] = pl['選手ID'].apply(normalize_player_id)
        pl = pl.drop_duplicates('選手ID').set_index('選手ID')
        p['速球'] = p.index.to_series().map(pl['球種データ'].apply(fastball_velocity)) if '球種データ' in pl else np.nan
        p['速球'] = pd.to_numeric(p['速球'], errors='coerce')
        p['投'] = p.index.to_series().map(pl['投'])
        p['球団'] = p.index.to_series().map(pl['球団'])
        p['選手名'] = p.index.to_series().map(pl['選手名']).fillna(p['選手名'])
        b['打'] = b.index.to_series().map(pl['打'])
        b['選手名'] = b.index.to_series().map(pl['選手名']).fillna(b['選手名'])
    else:
        p['速球'] = np.nan
        p['投'] = None
        p['球団'] = None
        b['打'] = None
    ref = p[p['bf'] >= MIN_PA_FOR_THRESHOLD]
    zk = _zscore(p['K%'], ref['K%'])
    zv = _zscore(p['速球'], ref['速球'])
    p['速球派'] = pd.concat([zk, zv], axis=1).mean(axis=1, skipna=True) >= 0
    if not use_style:   # 左右だけの相性（全員を「長打型」「速球派」に入れる）
        b['長打型'] = True
        p['速球派'] = True

    # --- 打席ごとのタイプと期待値 ---
    df['打者タイプ'] = batter_type_label(df['打者_打席'], df['打者ID'].map(b['長打型']))
    df['投手タイプ'] = pitcher_type_label(df['投手_腕'], df['投手ID'].map(p['速球派']))
    exp = df['打者ID'].map(b['w']) + df['投手ID'].map(p['w']) - lg
    df['resid'] = df['woba_num'] - df['woba_den'] * exp
    g = df.dropna(subset=['打者タイプ', '投手タイプ']).groupby(['打者タイプ', '投手タイプ'])
    cell = g['resid'].sum() / (g['woba_den'].sum() + c_match)
    matchup = cell.unstack().reindex(index=BATTER_TYPES, columns=PITCHER_TYPES).fillna(0.0)
    matchup_n = g['woba_den'].sum().unstack().reindex(index=BATTER_TYPES, columns=PITCHER_TYPES).fillna(0).astype(int)

    # --- 係数 k: 打席ごとの「wOBA の値 − lg」と ΔRE の関係（切片なし回帰）---
    #   収縮後の打者 wOBA に回帰すると、学習データ自身への当てはまりで k が大きく出すぎる（約1.2）ため、
    #   打席の結果そのもの（四球なら 0.70 − lg、凡退なら 0 − lg …）に回帰して「wOBA 1点 = 何点」を求める
    k = _fit_k(df, re24, df['woba_num'] - lg * df['woba_den'])

    return {'lg': lg, 'k': k, 'batters': b, 'pitchers': p, 'matchup': matchup, 'matchup_n': matchup_n,
            'use_style': use_style,
            'params': {'C_BAT': c_bat, 'C_PIT': c_pit, 'C_MATCH': c_match,
                       'ISO境目': float(iso_cut), 'lg_ISO': float(lg_iso), 'lg_K%': float(lg_k)}}


BATTER_TYPES  = ['右打・長打型', '右打・巧打型', '左打・長打型', '左打・巧打型']
PITCHER_TYPES = ['右投・速球派', '右投・技巧派', '左投・速球派', '左投・技巧派']


def batter_type_label(stance, power) -> pd.Series:
    """打席（右/左）と長打型フラグから '右打・長打型' のようなラベルを作る（どちらか不明なら None）"""
    s, pw = pd.Series(stance).reset_index(drop=True), pd.Series(power).reset_index(drop=True)
    lab = s.map({'右': '右打', '左': '左打'}) + '・' + pw.map({True: '長打型', False: '巧打型'})
    lab.index = pd.Series(stance).index
    return lab


def pitcher_type_label(arm, fast) -> pd.Series:
    """投げ腕（右/左）と速球派フラグから '右投・速球派' のようなラベルを作る（どちらか不明なら None）"""
    a, f = pd.Series(arm).reset_index(drop=True), pd.Series(fast).reset_index(drop=True)
    lab = a.map({'右': '右投', '左': '左投'}) + '・' + f.map({True: '速球派', False: '技巧派'})
    lab.index = pd.Series(arm).index
    return lab


def _fit_k(df: pd.DataFrame, re24: pd.DataFrame, x: pd.Series) -> float:
    """ΔRE = RE(次の打席) − RE(この打席) + この打席の得点 を x に回帰した係数（切片なし）"""
    d = df.copy()
    d['x'] = x
    d['runner_idx'] = d['ランナー'].fillna('').map(RUNNER_MAP).fillna(0).astype(int)
    arr = re24.values
    d['RE'] = [arr[o, r] if 0 <= o <= 2 else np.nan for o, r in zip(d['アウト'].astype(int), d['runner_idx'])]
    grp = d.groupby(['game_id', 'イニング'], sort=False)
    nxt_re, nxt_runs = grp['RE'].shift(-1), grp['runs_after'].shift(-1)
    d['dRE'] = nxt_re.fillna(0) - d['RE'] + (d['runs_after'] - nxt_runs.fillna(0))
    d = d.dropna(subset=['dRE', 'x'])
    den = float((d['x'] ** 2).sum())
    return float((d['x'] * d['dRE']).sum() / den) if den > 0 else 0.0


def type_features(df: pd.DataFrame, model: dict, df_players: pd.DataFrame | None) -> pd.DataFrame:
    """
    打席データ（打者ID・投手ID つき）の各打席に、相性モデルの要素を列で付ける。
      tm_打者  : 打者 wOBA（収縮後。学習に出ない打者は lg）
      tm_投手  : 投手補正 = 投手の被 wOBA（収縮後）− lg（学習に出ない投手は 0）
      tm_相性  : 相性[打者タイプ, 投手タイプ]（タイプが分からなければ 0）
      tm_打者タイプ, tm_投手タイプ
    """
    lg, b, p, mt = model['lg'], model['batters'], model['pitchers'], model['matchup']
    d = attach_player_info(df, df_players)
    out = pd.DataFrame(index=df.index)
    out['tm_打者'] = d['打者ID'].map(b['w']).fillna(lg)
    out['tm_投手'] = (d['投手ID'].map(p['w']) - lg).fillna(0.0)
    bt = batter_type_label(d['打者_打席'], d['打者ID'].map(b['長打型']))
    pt = pitcher_type_label(d['投手_腕'], d['投手ID'].map(p['速球派']))
    stacked = mt.stack()
    out['tm_相性'] = [float(stacked.get((x, y), 0.0)) if isinstance(x, str) and isinstance(y, str) else 0.0
                      for x, y in zip(bt, pt)]
    out['tm_打者タイプ'], out['tm_投手タイプ'] = bt, pt
    return out


def type_matchup_ability(model: dict, batter_id, pitcher_weights: dict | None = None,
                         batter_form: str = '普通', pitcher_form: str = '普通',
                         batter_w: float | None = None) -> dict:
    """
    打者の能力 w を、対戦投手（または投手の集まり）に合わせて計算する。
      pitcher_weights: {投手ID: 重み}。1人なら {ID: 1}。None なら投手を考えない（打者だけ）
    返り値: w, 打者wOBA, 投手補正, 相性補正, 打者タイプ（右/左投手それぞれ）, 内訳など
    """
    lg = model['lg']
    b, p, mt = model['batters'], model['pitchers'], model['matchup']
    bid = normalize_player_id(batter_id)
    if bid in b.index:
        brow = b.loc[bid]
        w_b, pa = float(brow['w']), int(brow['打席'])
        power, bats = bool(brow['長打型']), brow.get('打')
    else:
        w_b, pa, power, bats = lg, 0, None, None
    if batter_w is not None:   # 条件別 wOBA（conditional_batter_woba）で置き換える
        w_b = float(batter_w)
    # 調子: 好調なら打者 wOBA を FORM_RATE 割増し、不調なら割引き
    w_b *= 1 + FORM_RATE * FORM_SIGN.get(batter_form, 0)
    p_mult = 1 - FORM_RATE * FORM_SIGN.get(pitcher_form, 0)   # 投手が好調なら被 wOBA を割引き

    res = {'打者wOBA': w_b, '打者打席': pa, '投手補正': 0.0, '相性補正': 0.0,
           '打者タイプ': None, '投手タイプ': None, '投手': []}
    if not pitcher_weights:
        res['w'] = w_b
        res['打者タイプ'] = _type_for(bats, power, None)
        return res

    tot = sum(pitcher_weights.values())
    p_eff = m_eff = 0.0
    for pid, wt in pitcher_weights.items():
        pid = normalize_player_id(pid)
        wt = wt / tot if tot else 0
        prow = p.loc[pid] if pid in p.index else None
        pe = float(prow['w']) * p_mult - lg if prow is not None else lg * p_mult - lg
        arm = prow['投'] if prow is not None else None
        fast = bool(prow['速球派']) if prow is not None else None
        bt = _type_for(bats, power, arm)
        pt = (pitcher_type_label(pd.Series([arm]), pd.Series([fast])).iloc[0]
              if arm in ('右', '左') and fast is not None else None)
        me = float(mt.loc[bt, pt]) if bt in mt.index and pt in mt.columns else 0.0
        p_eff += wt * pe
        m_eff += wt * me
        res['投手'].append({'投手ID': pid, '重み': wt, '投手補正': pe, '相性補正': me,
                            '打者タイプ': bt, '投手タイプ': pt})
    res['投手補正'], res['相性補正'] = p_eff, m_eff
    res['w'] = w_b + p_eff + m_eff
    if len(res['投手']) == 1:
        res['打者タイプ'], res['投手タイプ'] = res['投手'][0]['打者タイプ'], res['投手'][0]['投手タイプ']
    else:
        types = {x['打者タイプ'] for x in res['投手'] if x['打者タイプ']}
        if len(types) == 1:
            res['打者タイプ'] = types.pop()
        elif bats == '両' and power is not None:
            res['打者タイプ'] = '両打・' + ('長打型' if power else '巧打型')
    return res


def _type_for(bats, power, pitcher_arm) -> str | None:
    """打者の打席（両打は投手の逆側）と長打型フラグからタイプ名"""
    if power is None or bats not in ('右', '左', '両'):
        return None
    stance = {'右': '左', '左': '右'}.get(pitcher_arm) if bats == '両' else bats
    if stance is None:
        stance = '左' if bats == '両' else bats   # 両打で投手不明なら、多い右投手を想定して左打席
    return batter_type_label(pd.Series([stance]), pd.Series([power])).iloc[0]


def team_pitcher_weights(model: dict, team: str) -> dict:
    """相手チームの投手を、学習データの対戦打者数で重み付けした {投手ID: 重み}（team は短縮名）"""
    p = model['pitchers']
    if '球団' not in p:
        return {}
    teams = p['球団'].apply(normalize_team_name)
    sub = p[teams == team]
    return {pid: float(n) for pid, n in sub['対戦打者'].items() if n > 0}


def predict_type_one(re24: pd.DataFrame, model: dict, batter_id, pitcher_weights: dict | None,
                     out: int, runner: str, batter_form: str = '普通', pitcher_form: str = '普通',
                     batter_w: float | None = None) -> dict:
    """1場面の予測（足し算の形）。補正後期待得点 = RE24 + k × (w − lg)。batter_w で打者 wOBA を置き換えられる"""
    base_re = re24.iloc[out, RUNNER_MAP.get(runner, 0)]
    a = type_matchup_ability(model, batter_id, pitcher_weights, batter_form, pitcher_form, batter_w)
    adj = base_re + model['k'] * (a['w'] - model['lg']) if not np.isnan(base_re) else np.nan
    return {'アウト': out, 'ランナー': runner if runner else '走者なし',
            '打者wOBA': round(a['打者wOBA'], 3), '投手補正': round(a['投手補正'], 3),
            '相性補正': round(a['相性補正'], 3), '対戦wOBA予測': round(a['w'], 3),
            '基礎RE24': round(float(base_re), 3) if not np.isnan(base_re) else None,
            '補正後期待得点': round(float(adj), 3) if not np.isnan(adj) else None,
            '打者タイプ': a['打者タイプ'], '投手タイプ': a['投手タイプ'], '打者打席': a['打者打席']}


def find_batter_id(model: dict | None, batter_name: str) -> str | None:
    """
    打者名（nf3 の表記）から相性モデルの打者IDを探す。同名の別人（オスナ、マルティネスなど）が
    いる場合は、打席の多い方（＝打者として出ている方）を選ぶ。
    """
    if model is None:
        return None
    b = model['batters']
    key = normalize_name(batter_name)
    hit = b[b['選手名'].apply(normalize_name) == key]
    return hit['打席'].idxmax() if not hit.empty else None


def predict_type_all(re24, model, batter_id, pitcher_weights,
                     batter_form: str = '普通', pitcher_form: str = '普通', batter_w_fn=None) -> pd.DataFrame:
    """全24場面。batter_w_fn(out, runner) を渡すと、場面ごとに打者 wOBA を置き換える（条件別 wOBA）"""
    return pd.DataFrame([predict_type_one(re24, model, batter_id, pitcher_weights, o, r, batter_form, pitcher_form,
                                          batter_w_fn(o, r) if batter_w_fn else None)
                         for o in range(3) for r in RUNNER_LABEL])


# ============================================================
# Step 9: 調子（好調・普通・不調）（2026-11-26）
#
#   判定: 直近の試合（打者 10 試合・投手 5 登板）の wOBA（投手は被 wOBA）が、
#         シーズンの実力（相性モデルの収縮後 wOBA）からどれだけ離れているかを z 値で見る
#           z = (直近の wOBA 分子 − 直近の分母 × シーズンの wOBA) / (WOBA_PA_SD × √直近の分母)
#         打者: z ≥ +1 で好調、z ≤ −1 で不調。投手は向きが逆（被 wOBA が低いほど好調）
#         直近の分母が FORM_MIN_PA 未満なら判定しない（普通）
#   効き方: 好調で打者 wOBA ×(1 + FORM_RATE)、不調で ×(1 − FORM_RATE)。投手は被 wOBA に逆向きで掛ける
#   アプリの既定は「普通」（判定結果は参考として表示し、選んだときだけ使う）
# ============================================================
FORM_LEVELS = ['不調', '普通', '好調']
FORM_SIGN = {'不調': -1, '普通': 0, '好調': 1}
FORM_RATE = 0.10          # 調子による割合補正（ユーザー指定 10〜20% の下限）
FORM_N_GAMES_BAT = 10     # 打者の「直近」= 出場した直近 10 試合
FORM_N_GAMES_PIT = 5      # 投手の「直近」= 直近 5 登板
FORM_MIN_PA = 15          # 判定に必要な最低打席数・対戦打者数
FORM_Z = 1.0              # 好調・不調の境目（z 値）
WOBA_PA_SD = 0.50         # 1打席の wOBA の値の標準偏差（2026年の一球速報で 0.496）


def build_form_table(df_runs: pd.DataFrame, model: dict, as_of_game_id=None) -> dict:
    """
    全打者・全投手の直近の調子を判定する。
      df_runs: calc_runs_after の出力（オールスターは除いて渡す。ポストシーズンは含めてよい）
      as_of_game_id: この試合ID より前の試合だけを見る（None なら全試合）
    Returns: {'打者': DataFrame(index=打者ID), '投手': DataFrame(index=投手ID)}
             列: 直近試合数, 直近打席, 直近wOBA, シーズンwOBA, z, 調子
    """
    df = add_woba_columns(add_result_columns(df_runs))
    for c in ('打者ID', '投手ID'):
        df[c] = df[c].apply(normalize_player_id)
    df['gid'] = df['game_id'].astype(int)
    if as_of_game_id is not None:
        df = df[df['gid'] < int(as_of_game_id)]
    out = {}
    for role, col, n_games, season, sign in (('打者', '打者ID', FORM_N_GAMES_BAT, model['batters']['w'], 1),
                                              ('投手', '投手ID', FORM_N_GAMES_PIT, model['pitchers']['w'], -1)):
        d = df[df[col].notna()]
        last = (d[['gid', col]].drop_duplicates().sort_values('gid')
                .groupby(col).tail(n_games))
        recent = d.merge(last, on=['gid', col])
        g = recent.groupby(col).agg(直近試合数=('gid', 'nunique'), num=('woba_num', 'sum'),
                                    直近打席=('woba_den', 'sum'), 最終試合=('gid', 'max'))
        g['直近wOBA'] = (g['num'] / g['直近打席']).where(g['直近打席'] > 0)
        g['シーズンwOBA'] = g.index.to_series().map(season).fillna(model['lg'])
        g['z'] = (g['num'] - g['直近打席'] * g['シーズンwOBA']) / (WOBA_PA_SD * np.sqrt(g['直近打席'].clip(lower=1)))
        zz = sign * g['z']
        g['調子'] = np.where(g['直近打席'] < FORM_MIN_PA, '普通',
                             np.where(zz >= FORM_Z, '好調', np.where(zz <= -FORM_Z, '不調', '普通')))
        out[role] = g.drop(columns=['num'])
    return out


def form_of(form_table: dict | None, role: str, player_id) -> dict | None:
    """判定結果の1行を dict で返す（データがなければ None）"""
    if not form_table:
        return None
    t = form_table.get(role)
    pid = normalize_player_id(player_id)
    if t is None or pid not in t.index:
        return None
    return t.loc[pid].to_dict()



# ============================================================
# Step 10: wOBA の係数の算出と、選手ごとの条件別 wOBA（2026-11-26）
# ============================================================
ERROR_TEXT = re.compile(r'失策|エラー|悪送球|ファンブル|後逸|落球|野選|フィルダースチョイス')


def compute_woba_weights(df_runs: pd.DataFrame, re24: pd.DataFrame) -> dict:
    """
    一球速報（レギュラーシーズン）から wOBA の係数を算出する（Tango の方法）。
      1. 各打席の得点価値 ΔRE = RE(次の打席の状況) − RE(この打席の状況) + この打席で入った得点
      2. 結果ごとに ΔRE を平均 → 得点価値（線形加重）
      3. 凡退（三振・凡打・犠飛。失策出塁は除く）の得点価値を引く
      4. リーグの wOBA がリーグの出塁率と同じになるように全体を定数倍（wOBA スケール）
    Returns: dict(weights=係数 {BB,HBP,S,D,T,HR}, scale, league_obp, out_value, table=結果ごとの得点価値の表)
    """
    d = add_result_columns(df_runs)
    d['runner_idx'] = d['ランナー'].fillna('').map(RUNNER_MAP).fillna(0).astype(int)
    arr = re24.values
    d['RE'] = [arr[o, r] for o, r in zip(d['アウト'].astype(int), d['runner_idx'])]
    g = d.groupby(['game_id', 'イニング'], sort=False)
    d['dRE'] = g['RE'].shift(-1).fillna(0) - d['RE'] + (d['runs_after'] - g['runs_after'].shift(-1).fillna(0))
    d['ev'] = d['結果']
    d.loc[(d['結果'] == 'OUT') & d['本文'].fillna('').str.contains(ERROR_TEXT), 'ev'] = 'ROE'
    table = d.groupby('ev')['dRE'].agg(得点価値='mean', 件数='size')
    out_value = float(d[d['ev'].isin(['OUT', 'SO', 'SF'])]['dRE'].mean())

    c = d['結果'].value_counts()
    n = {k: int(c.get(k, 0)) for k in ('1B', '2B', '3B', 'HR', 'BB', 'IBB', 'HBP', 'SF', 'SAC')}
    hits = n['1B'] + n['2B'] + n['3B'] + n['HR']
    ab = int((~d['結果'].isin(['BB', 'IBB', 'HBP', 'SF', 'SAC', 'UNKNOWN'])).sum())
    obp = (hits + n['BB'] + n['IBB'] + n['HBP']) / (ab + n['BB'] + n['IBB'] + n['HBP'] + n['SF'])
    key = {'BB': 'BB', 'HBP': 'HBP', 'S': '1B', 'D': '2B', 'T': '3B', 'HR': 'HR'}
    raw = {k: float(table.loc[e, '得点価値']) - out_value for k, e in key.items()}
    raw_num = sum(raw[k] * n[e] for k, e in key.items())
    scale = obp / (raw_num / (ab + n['BB'] + n['HBP'] + n['SF']))
    table['凡退との差'] = table['得点価値'] - out_value
    return {'weights': {k: v * scale for k, v in raw.items()}, 'scale': float(scale),
            'league_obp': float(obp), 'out_value': out_value, 'table': table}


RISP_BITS = {'2塁', '3塁', '1,2塁', '1,3塁', '2,3塁', '満塁'}   # 得点圏


def build_pa_table(df_runs: pd.DataFrame, df_players: pd.DataFrame | None, model: dict) -> pd.DataFrame:
    """
    条件別 wOBA 用の打席の表（calc_runs_after の出力。オールスターを除き、ポストシーズンは含めてよい）。
    列: gid, 区分, 打者ID, 打者名, 投手ID, 投手名, 打者球団, 投手球団, ホーム/ビジター, イニング番号, アウト, 走者,
        結果, 打者の打席, 投手の腕, 打者タイプ, 投手タイプ, woba_num, woba_den
    """
    d = attach_player_info(add_woba_columns(add_result_columns(df_runs)), df_players)
    d = d[d['結果'] != 'UNKNOWN']
    side = np.where(d['イニング'].astype(str).str.contains('表'), 'ビジター', 'ホーム')
    bat_raw = np.where(side == 'ビジター', d['away_team_raw'], d['home_team_raw'])
    teams = {t: normalize_team_name(t) for t in pd.unique(np.concatenate([d['away_team_raw'].dropna().unique(),
                                                                       d['home_team_raw'].dropna().unique()]))}
    names = {}
    if df_players is not None and not df_players.empty:
        names = dict(zip(df_players['選手ID'].apply(normalize_player_id), df_players['選手名']))
    gid = d['game_id'].astype(int)
    out = pd.DataFrame({
        'gid': gid,
        '区分': np.where(gid.isin(POSTSEASON_IDS), 'ポストシーズン', 'レギュラーシーズン'),
        '打者ID': d['打者ID'], '投手ID': d['投手ID'],
        '打者名': d['打者ID'].map(names).fillna(d['選手名']),
        '投手名': d['投手ID'].map(names).fillna(d['投手名'] if '投手名' in d else None),
        '打者球団': pd.Series(bat_raw, index=d.index).map(teams),
        '投手球団': d['対戦球団'],
        'ホーム/ビジター': side,
        'イニング番号': d['イニング'].map(lambda x: _inning_sort_key(x)[0]),
        'アウト': d['アウト'].astype(int),
        '走者': d['ランナー'].fillna('').replace('', '走者なし'),
        '結果': d['結果'],
        '打者の打席': d['打者_打席'], '投手の腕': d['投手_腕'],
        '打者タイプ': batter_type_label(d['打者_打席'], d['打者ID'].map(model['batters']['長打型'])),
        '投手タイプ': pitcher_type_label(d['投手_腕'], d['投手ID'].map(model['pitchers']['速球派'])),
        'woba_num': d['woba_num'], 'woba_den': d['woba_den'],
    })
    return out.reset_index(drop=True)


def woba_stats(pa: pd.DataFrame) -> dict:
    """打席の表（build_pa_table の行）から打撃成績と wOBA を計算する"""
    c = pa['結果'].value_counts()
    n = {k: int(c.get(k, 0)) for k in ('1B', '2B', '3B', 'HR', 'BB', 'IBB', 'HBP', 'SO', 'SF', 'SAC')}
    h = n['1B'] + n['2B'] + n['3B'] + n['HR']
    tpa = len(pa)
    ab = tpa - n['BB'] - n['IBB'] - n['HBP'] - n['SF'] - n['SAC']
    obp_den = ab + n['BB'] + n['IBB'] + n['HBP'] + n['SF']
    den = pa['woba_den'].sum()
    return {'打席': tpa, '打数': ab, '安打': h, '二塁打': n['2B'], '三塁打': n['3B'], '本塁打': n['HR'],
            '四球': n['BB'] + n['IBB'], '敬遠': n['IBB'], '死球': n['HBP'], '三振': n['SO'],
            '犠打': n['SAC'], '犠飛': n['SF'],
            '打率': h / ab if ab else np.nan,
            '出塁率': (h + n['BB'] + n['IBB'] + n['HBP']) / obp_den if obp_den else np.nan,
            '長打率': (n['1B'] + 2 * n['2B'] + 3 * n['3B'] + 4 * n['HR']) / ab if ab else np.nan,
            'wOBA': pa['woba_num'].sum() / den if den else np.nan}


def woba_breakdown(pa: pd.DataFrame, by: str) -> pd.DataFrame:
    """条件（列名 by）ごとの打撃成績と wOBA の表"""
    rows = []
    for key, g in pa.groupby(by, dropna=False):
        r = {by: key if pd.notna(key) else '不明'}
        r.update(woba_stats(g))
        rows.append(r)
    return pd.DataFrame(rows)



# ============================================================
# Step 11: メインの予測で使う条件別 wOBA（2026-11-26）
#   打者の wOBA を「いまの条件」（走者・アウト・対戦球団・ホーム/ビジター など）での成績に置き換える。
#   条件で絞ると打席が少なくなるので、打者の全体 wOBA（相性モデルの収縮後）へ C_COND 打席分収縮する。
#     条件別 wOBA = (条件での wOBA 分子 + C_COND × 全体 wOBA) / (条件での分母 + C_COND)
#   検証（validate.py）では、走者・得点圏・アウト・対戦球団の条件を入れても予測はほぼ変わらない
#   （C=200 で RE24 だけとの差の改善は ±0.0002 以内）。ホーム/ビジターはポストシーズンでわずかに改善。
# ============================================================
C_COND = 200
COND_COLUMNS = {'走者': '走者', '得点圏': '得点圏', 'アウト': 'アウト', '対戦球団': '投手球団',
                'ホーム/ビジター': 'ホーム/ビジター'}


def conditional_batter_woba(model: dict, batter_id, conds: dict, c: float = C_COND) -> dict:
    """
    conds: {'走者': '1,2塁', 'アウト': 1, '対戦球団': '巨人', 'ホーム/ビジター': 'ホーム', '得点圏': True} の一部
    レギュラーシーズンの打席だけを使う。Returns: w（収縮後）, 生の wOBA, 打席, 全体 wOBA
    """
    pa = model.get('pa')
    bid = normalize_player_id(batter_id)
    base = float(model['batters']['w'].get(bid, model['lg']))
    if pa is None or bid is None:
        return {'w': base, 'wOBA': np.nan, '打席': 0, '全体': base}
    d = pa[(pa['打者ID'] == bid) & (pa['区分'] == 'レギュラーシーズン')]
    for k, v in conds.items():
        if k == '得点圏':
            d = d[d['走者'].isin(RISP_BITS) == bool(v)]
        else:
            d = d[d[COND_COLUMNS[k]] == v]
    num, den = float(d['woba_num'].sum()), float(d['woba_den'].sum())
    return {'w': (num + c * base) / (den + c), 'wOBA': num / den if den else np.nan,
            '打席': int(den), '全体': base}
