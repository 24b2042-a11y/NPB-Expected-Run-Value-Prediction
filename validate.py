"""
validate.py — 得点期待値予測の検証（学習データとテストデータに分けて、予測と実測を比較する）

検証設計（研究計画 2026-11-19）
    学習: 2026年レギュラーシーズン（オールスターは除く）
    テスト: 2026年ポストシーズン（クライマックスシリーズ＋日本シリーズ、16試合）
    補助: レギュラーシーズン内の時系列分割（--split season-tail）

目的変数（実測値）は2通りを両方出す（どちらを本命にするかは未決定）
    runs_after : その打席からイニング終了までに入った得点（RE24 と同じ定義。後続打者の分も含む）
    dRE        : その打席の得点価値 ΔRE = RE(打席後の状況) − RE(打席前の状況) + その打席で入った得点
                 RE は学習データの RE24。イニング最後の打席は RE(打席後) = 0

予測の形（打者能力 w = その打者の wOBA 予測、lg = リーグ平均 wOBA）
    乗法型（現在のアプリの式）: runs_after = RE × (w / lg)、        dRE = RE × (w / lg − 1)
    加法型（wRAA 型）        : runs_after = RE + k × (w − lg)、     dRE = k × (w − lg)
        k は学習データで dRE を (w − lg) に回帰して求める（切片なし。打者全体 wOBA を使用）

打者能力 w の作り方
    lg          : 全員リーグ平均（＝ RE24 だけのベースライン）
    全体        : その打者の全体 wOBA（全対戦相手の合計）
    全体収縮    : 全体 wOBA をリーグ平均へ収縮（C_lg 打席分）
    対戦生      : 対戦相手別 wOBA（分母 10 未満なら全体 wOBA。以前のアプリの方式）
    対戦収縮C   : 対戦相手別 wOBA を全体 wOBA へ収縮（core.predict_batter_woba、C を変える）
    対戦収縮C+全体収縮 : 二段階（全体 wOBA をリーグ平均へ収縮してから、対戦 wOBA をそこへ収縮）

wOBA の元データ（--woba-source）
    pbp : 学習期間の一球速報から計算（既定。12球団そろっていて、テスト期間を含まない）
    nf3 : data/all_batters_situational.csv（レギュラーシーズンの対戦相手別成績）
          ※ season-tail 分割ではテスト期間の打席を含むので使えない（自動で pbp にする）
    pbp : 学習期間の一球速報から計算（core.classify_result の打席結果コードから wOBA を計算）

評価指標
    RMSE, MAE, 平均誤差（予測 − 実測）
    ベースライン（RE24 だけ）との RMSE の差と、試合単位のブートストラップによる 95% 区間
    キャリブレーション（予測値の大きさで 10 分割し、各区間の予測平均と実測平均）

使い方
    python validate.py                                # 学習=シーズン、テスト=ポストシーズン
    python validate.py --split season-tail --test-frac 0.2
    python validate.py --c-list 5,10,20,50,100 --n-boot 2000
結果は results/ に CSV で保存し、要約を画面に表示する。
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import core
from core import (RUNNER_MAP, WOBA_W, normalize_name, concat_details, calc_runs_after,
                  add_result_columns, build_re24, build_batter_woba)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GAMEDATA_DIR = os.path.join(BASE_DIR, 'data', 'gamedata')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# オールスター・ポストシーズンの試合IDは core.py に定義（アプリと共通）
POSTSEASON_IDS = core.POSTSEASON_IDS
ALLSTAR_IDS = core.ALLSTAR_IDS

DEFAULT_C_LIST = [5, 10, 20, 50, 100, 200]
C_LEAGUE = 100   # 全体 wOBA をリーグ平均へ収縮するときの強さ（打席数）


# ============================================================
# データ読み込み
# ============================================================
def load_details_local(data_dir: str = GAMEDATA_DIR) -> list[pd.DataFrame]:
    """
    data/gamedata/*_details.csv を読み込む（github_sync.load_details_from_github のローカル版）。
    ファイル名 "{game_id}_{Home}vs.{Away}_details.csv" から game_id とホーム/アウェイ球団名を付ける。
    """
    dfs = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*_details.csv'))):
        fname = os.path.basename(path)
        df = pd.read_csv(path, encoding='utf-8-sig')
        df['game_id'] = fname.split('_', 1)[0]
        card = fname.split('_', 1)[1].replace('_details.csv', '')
        home_raw, away_raw = card.split('vs.', 1) if 'vs.' in card else (None, None)
        df['home_team_raw'] = home_raw
        df['away_team_raw'] = away_raw
        dfs.append(df)
    return dfs


def split_games(game_ids: list[int], split: str, test_frac: float) -> tuple[set, set]:
    """学習・テストの試合IDに分ける"""
    regular = sorted(g for g in game_ids if g not in POSTSEASON_IDS and g not in ALLSTAR_IDS)
    if split == 'postseason':
        test = {g for g in game_ids if g in POSTSEASON_IDS}
        return set(regular), test
    if split == 'season-tail':
        # 試合IDはおおむね日付順なので、ID の大きい方 test_frac をテストにする
        n_test = int(round(len(regular) * test_frac))
        return set(regular[:-n_test]), set(regular[-n_test:])
    raise ValueError(f'unknown split: {split}')


# ============================================================
# 目的変数
# ============================================================
def add_targets(df_runs: pd.DataFrame, re24: pd.DataFrame) -> pd.DataFrame:
    """
    RE（打席前の状況の期待得点）と ΔRE を付ける。RE は学習データの RE24 を使う。
    ΔRE = RE(次の打席の状況) − RE(この打席の状況) + この打席で入った得点
      この打席で入った得点 = runs_after(この打席) − runs_after(次の打席)
      イニング最後の打席: RE(次) = 0、入った得点 = runs_after(この打席)
    """
    df = df_runs.copy()
    df['game_id'] = df['game_id'].astype(str)
    df['_order'] = range(len(df))   # calc_runs_after は半イニングごとに打席順で並べて出力している
    df['runner_idx'] = df['ランナー'].fillna('').map(RUNNER_MAP).fillna(0).astype(int)
    re_arr = re24.values
    df['RE'] = [re_arr[o, r] if 0 <= o <= 2 else np.nan
                for o, r in zip(df['アウト'].astype(int), df['runner_idx'])]

    grp = df.groupby(['game_id', 'イニング'], sort=False)
    next_re   = grp['RE'].shift(-1)
    next_runs = grp['runs_after'].shift(-1)
    df['runs_on_play'] = df['runs_after'] - next_runs.fillna(0)
    df['dRE'] = next_re.fillna(0) - df['RE'] + df['runs_on_play']
    return df


# ============================================================
# 打者 wOBA テーブル
# ============================================================
def build_woba_from_pbp(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    学習期間の打席データから、打者×対戦球団の wOBA テーブルを作る（build_batter_woba と同じ列）。
      分子 = 0.70×四球(敬遠除く) + 0.73×死球 + 0.89×単打 + 1.27×二塁打 + 1.61×三塁打 + 2.10×本塁打
      分母 = 打席 − 敬遠 − 犠打（打席 = 結果を判定できた行。UNKNOWN は除く）
    """
    df = df_train[df_train['結果'] != 'UNKNOWN'].copy()
    w = {'BB': WOBA_W['BB'], 'HBP': WOBA_W['HBP'], '1B': WOBA_W['S'],
         '2B': WOBA_W['D'], '3B': WOBA_W['T'], 'HR': WOBA_W['HR']}
    df['num'] = df['結果'].map(w).fillna(0.0)
    df['den'] = (~df['結果'].isin(['IBB', 'SAC'])).astype(int)
    g = (df.groupby(['batter_key', '対戦球団'])
           .agg(選手名=('選手名', 'first'), 試合=('game_id', 'nunique'), 打席=('結果', 'size'),
                wOBA_num=('num', 'sum'), wOBA_den=('den', 'sum'))
           .reset_index()
           .rename(columns={'batter_key': '選手名_key', '対戦球団': '区分名'}))
    g['球団'] = None
    g['wOBA'] = np.where(g['wOBA_den'] >= 10, g['wOBA_num'] / g['wOBA_den'].clip(lower=1), np.nan)
    return g


def league_woba(df_woba: pd.DataFrame) -> float:
    """リーグ平均 wOBA（打席加重 = 分子の合計 / 分母の合計）"""
    den = df_woba['wOBA_den'].clip(lower=0).sum()
    return float(df_woba['wOBA_num'].sum() / den) if den > 0 else np.nan


# ============================================================
# 打者能力 w の予測（各方式）
# ============================================================
def batter_abilities(df_woba: pd.DataFrame, lg: float, c_list: list[float]) -> dict:
    """
    打者×対戦球団ごとの w を、方式別に辞書で返す。
    返り値: {方式名: {(選手名_key, 区分名): w}}, {方式名: {選手名_key: w}}（対戦データがない時の値）
    """
    t = df_woba.copy()
    t['wOBA_den'] = t['wOBA_den'].clip(lower=0)
    tot = t.groupby('選手名_key')[['wOBA_num', 'wOBA_den']].sum()
    overall = (tot['wOBA_num'] / tot['wOBA_den']).where(tot['wOBA_den'] > 0)
    overall_shr = (tot['wOBA_num'] + C_LEAGUE * lg) / (tot['wOBA_den'] + C_LEAGUE)

    t = t.join(overall.rename('全体'), on='選手名_key').join(overall_shr.rename('全体収縮'), on='選手名_key')
    opp_raw = np.where(t['wOBA_den'] >= 10, t['wOBA_num'] / t['wOBA_den'].clip(lower=1), t['全体'])

    pair, single = {}, {}
    key = list(zip(t['選手名_key'], t['区分名']))
    pair['全体']     = dict(zip(key, t['全体']))
    pair['全体収縮'] = dict(zip(key, t['全体収縮']))
    pair['対戦生']   = dict(zip(key, opp_raw))
    single['全体'] = single['対戦生'] = overall.to_dict()
    single['全体収縮'] = overall_shr.to_dict()
    for c in c_list:
        # core.predict_batter_woba と同じ式: (分子_対戦 + C×全体) / (分母_対戦 + C)
        pair[f'対戦収縮C{c:g}'] = dict(zip(key, (t['wOBA_num'] + c * t['全体']) / (t['wOBA_den'] + c)))
        single[f'対戦収縮C{c:g}'] = overall.to_dict()
        pair[f'対戦収縮C{c:g}+全体収縮'] = dict(zip(key, (t['wOBA_num'] + c * t['全体収縮']) / (t['wOBA_den'] + c)))
        single[f'対戦収縮C{c:g}+全体収縮'] = overall_shr.to_dict()
    return pair, single


def attach_ability(df: pd.DataFrame, pair: dict, single: dict, lg: float) -> pd.DataFrame:
    """テスト（または学習）の各打席に、方式別の w を列として付ける（w_方式名）"""
    out = df.copy()
    keys = list(zip(out['batter_key'], out['対戦球団']))
    out['w_lg'] = lg
    for name in pair:
        p, s = pair[name], single[name]
        vals = []
        for (b, o) in keys:
            v = p.get((b, o))
            if v is None or pd.isna(v):
                v = s.get(b)
            vals.append(lg if v is None or pd.isna(v) else float(v))
        out[f'w_{name}'] = vals
    return out


def fit_k(df_train: pd.DataFrame, lg: float) -> float:
    """加法型の係数 k: dRE ≈ k × (w全体 − lg) を学習データで最小二乗（切片なし）"""
    d = df_train.dropna(subset=['dRE', 'w_全体'])
    x = d['w_全体'] - lg
    return float((x * d['dRE']).sum() / (x * x).sum()) if (x * x).sum() > 0 else 0.0


# ============================================================
# 予測と評価
# ============================================================
def make_predictions(df: pd.DataFrame, lg: float, k: float, target: str) -> pd.DataFrame:
    """モデル名 → 予測値 の列を持つ DataFrame を返す"""
    preds = {}
    wcols = [c for c in df.columns if c.startswith('w_') and c != 'w_lg']
    re_ = df['RE']
    base = re_ if target == 'runs_after' else 0.0 * re_
    preds['RE24のみ'] = base
    for wc in wcols:
        name = wc[2:]
        r = df[wc] / lg
        if target == 'runs_after':
            preds[f'乗法:{name}'] = re_ * r
            preds[f'加法:{name}'] = re_ + k * (df[wc] - lg)
        else:
            preds[f'乗法:{name}'] = re_ * (r - 1)
            preds[f'加法:{name}'] = k * (df[wc] - lg)
    return pd.DataFrame(preds, index=df.index)


def evaluate(y: pd.Series, preds: pd.DataFrame, games: pd.Series, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """RMSE・MAE・平均誤差と、ベースライン（RE24のみ）との RMSE 差の試合単位ブートストラップ区間"""
    rng = np.random.default_rng(seed)
    err2 = preds.sub(y, axis=0) ** 2
    # 試合ごとの二乗誤差の合計と打席数（ブートストラップを速くするため先に集計）
    g_sum = err2.groupby(games.values).sum()
    g_n   = err2.groupby(games.values).size()
    gids  = g_sum.index.to_numpy()
    boot_idx = rng.integers(0, len(gids), size=(n_boot, len(gids)))

    rows = []
    base = 'RE24のみ'
    base_rmse_b = np.sqrt(g_sum[base].to_numpy()[boot_idx].sum(1) / g_n.to_numpy()[boot_idx].sum(1))
    for m in preds.columns:
        e = preds[m] - y
        rmse = float(np.sqrt((e ** 2).mean()))
        rmse_b = np.sqrt(g_sum[m].to_numpy()[boot_idx].sum(1) / g_n.to_numpy()[boot_idx].sum(1))
        diff_b = rmse_b - base_rmse_b
        rows.append({'モデル': m, 'RMSE': rmse, 'MAE': float(e.abs().mean()), '平均誤差': float(e.mean()),
                     'RMSE差_vsRE24': rmse - float(np.sqrt(((preds[base] - y) ** 2).mean())),
                     '差_95%下限': float(np.percentile(diff_b, 2.5)),
                     '差_95%上限': float(np.percentile(diff_b, 97.5)),
                     '改善する割合': float((diff_b < 0).mean())})
        # 相性の上乗せ効果を見るため「打者+投手」との対応のある差も出す
        base2 = '加法:型:打者+投手'
        if base2 in preds.columns:
            b2 = np.sqrt(g_sum[base2].to_numpy()[boot_idx].sum(1) / g_n.to_numpy()[boot_idx].sum(1))
            d2 = rmse_b - b2
            rows[-1].update({'差_vs打者+投手': rmse - float(np.sqrt(((preds[base2] - y) ** 2).mean())),
                             '差2_95%下限': float(np.percentile(d2, 2.5)),
                             '差2_95%上限': float(np.percentile(d2, 97.5))})
    return pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)


def calibration(y: pd.Series, pred: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """予測値の大きさで n_bins 分割し、各区間の予測平均・実測平均・件数"""
    d = pd.DataFrame({'pred': pred, 'y': y}).dropna()
    d['bin'] = pd.qcut(d['pred'].rank(method='first'), n_bins, labels=False)
    return d.groupby('bin').agg(予測平均=('pred', 'mean'), 実測平均=('y', 'mean'), 件数=('y', 'size')).reset_index()


# ============================================================
# メイン
# ============================================================
def run(split: str = 'postseason', test_frac: float = 0.2, woba_source: str = 'pbp',
        c_list: list[float] | None = None, n_boot: int = 2000) -> dict:
    c_list = c_list or DEFAULT_C_LIST
    if split == 'season-tail' and woba_source == 'nf3':
        print('※ season-tail 分割では nf3 の成績がテスト期間を含むため、wOBA は pbp（学習期間の打席）から計算します')
        woba_source = 'pbp'

    dfs = load_details_local()
    df_runs = add_result_columns(calc_runs_after(concat_details(dfs)))
    df_runs['game_id_int'] = df_runs['game_id'].astype(int)

    # 打者の名前キー: 選手マスタ（打者ID → フルネーム）を優先、なければ一球速報の表記
    # ※ nf3 の成績（--woba-source nf3）は名前でしか結合できないので名前キーを使う。
    #   pbp のときは同名の別人（オスナ、マルティネス、ロドリゲス）を区別するため、後で打者IDに置き換える
    df_runs['batter_key'] = df_runs['選手名'].apply(normalize_name)
    players_path = os.path.join(BASE_DIR, 'data', 'players.csv')
    if os.path.exists(players_path) and '打者ID' in df_runs.columns:
        pl = pd.read_csv(players_path, encoding='utf-8-sig', dtype={'選手ID': str})
        id2key = dict(zip(pl['選手ID'], pl['選手名_key']))
        ids = df_runs['打者ID'].apply(core.normalize_player_id)
        df_runs['batter_key'] = ids.map(id2key).fillna(df_runs['batter_key'])

    if woba_source == 'pbp' and '打者ID' in df_runs.columns:
        df_runs['batter_key'] = df_runs['打者ID'].apply(core.normalize_player_id).fillna(df_runs['batter_key'])

    train_ids, test_ids = split_games(sorted(df_runs['game_id_int'].unique()), split, test_frac)
    if not test_ids:
        raise SystemExit('テストの試合がありません。ポストシーズンを取得してください: '
                         'python scraper.py --ids 2021039474-2021039495')
    df_train = df_runs[df_runs['game_id_int'].isin(train_ids)]
    df_test  = df_runs[df_runs['game_id_int'].isin(test_ids)]

    re24, counts = build_re24(df_train)
    df_train = add_targets(df_train, re24)
    df_test  = add_targets(df_test, re24)

    if woba_source == 'nf3':
        df_woba, _ = build_batter_woba(pd.read_csv(os.path.join(BASE_DIR, 'data', 'all_batters_situational.csv'),
                                                   encoding='utf-8-sig'))
    else:
        df_woba = build_woba_from_pbp(df_train)
    lg = league_woba(df_woba)
    pair, single = batter_abilities(df_woba, lg, c_list)
    df_train = attach_ability(df_train, pair, single, lg)
    df_test  = attach_ability(df_test, pair, single, lg)
    k = fit_k(df_train, lg)

    # --- 打者タイプ × 投手タイプの相性モデル（core.build_type_model、学習データだけで作る）---
    if os.path.exists(players_path):
        pl_all = pd.read_csv(players_path, encoding='utf-8-sig', dtype={'選手ID': str, '背番号': str})
        variants = {
            '': dict(),                                  # 既定: C_PIT=100, C_MATCH=300
            '[左右のみ]': dict(use_style=False),
            '[C_PIT300]': dict(c_pit=300),
            '[C_MATCH100]': dict(c_match=100),
            '[C_MATCH1000]': dict(c_match=1000),
        }
        for tag, kw in variants.items():
            tm = core.build_type_model(df_train, pl_all, re24, **kw)
            for d in (df_train, df_test):
                f = core.type_features(d, tm, pl_all)
                if tag == '':
                    d['w_型:打者'] = f['tm_打者']
                    d['w_型:打者+投手'] = f['tm_打者'] + f['tm_投手']
                    d['w_型:打者+相性'] = f['tm_打者'] + f['tm_相性']
                    d['w_型:打者+投手+相性'] = f['tm_打者'] + f['tm_投手'] + f['tm_相性']
                elif tag == '[左右のみ]':
                    d['w_型:打者+投手+左右'] = f['tm_打者'] + f['tm_投手'] + f['tm_相性']
                else:
                    d[f'w_型:打者+投手+相性{tag}'] = f['tm_打者'] + f['tm_投手'] + f['tm_相性']

    known = set(df_woba['選手名_key'])
    match_rate = float(df_test['batter_key'].isin(known).mean())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {'学習試合': len(train_ids), 'テスト試合': len(test_ids),
           '学習打席': len(df_train), 'テスト打席': len(df_test),
           'リーグ平均wOBA': lg, 'k': k, '打者の一致率': match_rate, 'wOBA元': woba_source}
    for target in ('runs_after', 'dRE'):
        d = df_test.dropna(subset=['RE', target])
        preds = make_predictions(d, lg, k, target)
        res = evaluate(d[target], preds, d['game_id'], n_boot)
        res.to_csv(os.path.join(RESULTS_DIR, f'validation_{split}_{target}.csv'),
                   index=False, encoding='utf-8-sig')
        best = res[res['モデル'] != 'RE24のみ'].iloc[0]['モデル']
        cal = pd.concat([calibration(d[target], preds['RE24のみ']).assign(モデル='RE24のみ'),
                         calibration(d[target], preds[best]).assign(モデル=best)])
        cal.to_csv(os.path.join(RESULTS_DIR, f'calibration_{split}_{target}.csv'),
                   index=False, encoding='utf-8-sig')
        out[target] = res
    pd.DataFrame([{k_: v for k_, v in out.items() if not isinstance(v, pd.DataFrame)}]).to_csv(
        os.path.join(RESULTS_DIR, f'validation_{split}_summary.csv'), index=False, encoding='utf-8-sig')
    return out


def validate_simulation(n_sims: int = 300, n_regular: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    試合シミュレーション（game_sim.py）の検証。
      モデル（打者・投手の割合、相性、進塁の分布）はレギュラーシーズンで作り、
      ポストシーズン16試合を実際のスタメン・先発でそれぞれ n_sims 回シミュレーションして実際の結果と比べる。
      参考として、レギュラーシーズンから n_regular 試合を選んで同じことをする（学習データ内なので当てはまりの確認）。
    指標: 予測平均得点と実際の得点の差、勝敗のブライアスコア（ホームの勝率予測 vs 実際。引き分けは 0.5）
          ブライアスコアの基準 = 全試合で「ホーム勝率 = レギュラーシーズンのホーム勝率」と予測した場合
    """
    import game_sim
    dfs = load_details_local()
    runs = core.calc_runs_after(core.concat_details(dfs))
    gid = runs['game_id'].astype(int)
    reg = runs[~gid.isin(POSTSEASON_IDS | ALLSTAR_IDS)]
    re24, _ = build_re24(reg)
    pl = pd.read_csv(os.path.join(BASE_DIR, 'data', 'players.csv'), encoding='utf-8-sig',
                     dtype={'選手ID': str, '背番号': str})
    tm = core.build_type_model(reg, pl, re24)
    sim = game_sim.build_sim_model(reg, pl, tm)

    finals = runs.groupby('game_id').apply(_final_score, include_groups=False)
    reg_finals = finals[~finals.index.astype(int).isin(POSTSEASON_IDS | ALLSTAR_IDS)]
    home_rate = float(((reg_finals['home'] > reg_finals['away']) + 0.5 * (reg_finals['home'] == reg_finals['away'])).mean())

    rng = np.random.default_rng(seed)
    reg_ids = sorted(reg['game_id'].unique())
    pick = list(rng.choice(reg_ids, size=min(n_regular, len(reg_ids)), replace=False))
    rows = []
    for label, ids in (('ポストシーズン', sorted(str(g) for g in POSTSEASON_IDS if str(g) in finals.index)),
                       ('レギュラーシーズン（学習内）', pick)):
        for g in ids:
            setup = game_sim.game_setup(runs[runs['game_id'] == g])
            r = game_sim.simulate_many(sim, setup['away'], setup['home'], n=n_sims, seed=int(g) % 100000)
            f = finals.loc[g]
            actual = 1.0 if f['home'] > f['away'] else 0.0 if f['home'] < f['away'] else 0.5
            p_home = r['勝率']['home'] + 0.5 * r['勝率']['引き分け']
            rows.append({'区分': label, 'game_id': g, '予測_away': r['平均得点']['away'],
                         '予測_home': r['平均得点']['home'], '実際_away': f['away'], '実際_home': f['home'],
                         '予測ホーム勝率': p_home, '実際': actual,
                         'brier': (p_home - actual) ** 2, 'brier_基準': (home_rate - actual) ** 2})
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, 'simulation_check.csv'), index=False, encoding='utf-8-sig')
    return df


def _final_score(g: pd.DataFrame) -> pd.Series:
    """試合の最終スコア（半イニングごとの得点 = 最初の打席の runs_after を合計）"""
    first = g.groupby('イニング', sort=False).head(1)
    side = np.where(first['イニング'].astype(str).str.contains('表'), 'away', 'home')
    return pd.Series({'away': int(first.loc[side == 'away', 'runs_after'].sum()),
                      'home': int(first.loc[side == 'home', 'runs_after'].sum())})


def _print(out: dict, top: int) -> None:
    print(f"学習 {out['学習試合']} 試合 / {out['学習打席']:,} 打席　テスト {out['テスト試合']} 試合 / {out['テスト打席']:,} 打席")
    print(f"wOBA の元データ: {out['wOBA元']}　リーグ平均 wOBA {out['リーグ平均wOBA']:.3f}　"
          f"加法型の係数 k {out['k']:.3f}　テスト打席の打者が wOBA 表に見つかる割合 {out['打者の一致率']:.1%}")
    cols = ['モデル', 'RMSE', 'RMSE差_vsRE24', '差_95%下限', '差_95%上限', '改善する割合', '平均誤差']
    for target in ('runs_after', 'dRE'):
        res = out[target]
        print(f'\n=== 目的変数: {target}（RMSE の小さい順、上位 {top}。差がマイナス = RE24 だけより良い）===')
        show = pd.concat([res.head(top), res[res['モデル'] == 'RE24のみ']]).drop_duplicates('モデル')
        print(show[cols].to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='得点期待値予測の検証')
    ap.add_argument('--split', choices=['postseason', 'season-tail'], default='postseason')
    ap.add_argument('--test-frac', type=float, default=0.2, help='season-tail でテストにする割合')
    ap.add_argument('--woba-source', choices=['nf3', 'pbp'], default='pbp')
    ap.add_argument('--c-list', type=str, default=','.join(str(c) for c in DEFAULT_C_LIST))
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--sim', action='store_true', help='試合シミュレーション（game_sim.py）を検証する')
    ap.add_argument('--n-sims', type=int, default=300)
    args = ap.parse_args()
    if args.sim:
        df = validate_simulation(args.n_sims)
        for lab, d in df.groupby('区分'):
            pred = (d['予測_away'] + d['予測_home']).mean()
            act = (d['実際_away'] + d['実際_home']).mean()
            print(f"{lab}: {len(d)} 試合　1試合の両軍合計得点 予測 {pred:.2f} / 実際 {act:.2f}　"
                  f"ブライアスコア {d['brier'].mean():.3f}（基準 {d['brier_基準'].mean():.3f}。小さいほど良い）")
        raise SystemExit(0)
    o = run(args.split, args.test_frac, args.woba_source,
            [float(c) for c in args.c_list.split(',')], args.n_boot)
    _print(o, args.top)
