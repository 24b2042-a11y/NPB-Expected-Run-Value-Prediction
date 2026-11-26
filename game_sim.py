"""
game_sim.py — スタメンと先発投手から1試合をシミュレーションする（モンテカルロ法）

流れ
  1. 打席ごとの結果の確率（7種類: 三振・凡退・四死球・単打・二塁打・三塁打・本塁打）
       打者の割合と投手の割合をリーグ平均へ収縮（C = 100 打席）し、オッズ比法（一般化 log5）で合成
         p(結果) ∝ 打者の割合 × 投手の割合 ÷ リーグの割合      （Haechrel 2014）
       さらに打者タイプ × 投手タイプの相性（16マス × 7結果の倍率）と、調子（好調・不調）を掛ける
  2. 走者・アウトの進み方
       一球速報の「この打席の状況 → 次の打席の状況・入った得点」を、状況（24通り）× 結果（7種類）ごとに
       数えた経験分布から引く（併殺・進塁打・タッチアップ・盗塁などもまとめて入る）
       数が少ない組み合わせ（10回未満）は単純な進塁ルールで代用
  3. 投手交代: 先発は今季の先発登板の対戦打者数（その投手の登板からランダムに1つ）で降板し、
       そのあとはチームの救援投手全体（今季の救援での対戦打者数で重み付け）が投げる
  4. 9回まで。9回裏以降はホームがリードした時点で終了（サヨナラ）。12回で同点なら引き分け

計算は core.py の一球速報の処理（calc_runs_after, add_result_columns, attach_player_info）と、
相性モデル（build_type_model）のタイプ判定を使う。
"""
import numpy as np
import pandas as pd

from core import (RUNNER_MAP, add_result_columns, attach_player_info, normalize_player_id,
                  normalize_team_name, batter_type_label, pitcher_type_label, FORM_RATE, FORM_SIGN)

EVENTS = ['SO', 'OUT', 'BB', '1B', '2B', '3B', 'HR']
EVENT_LABEL = {'SO': '三振', 'OUT': '凡退', 'BB': '四死球', '1B': '単打', '2B': '二塁打', '3B': '三塁打', 'HR': '本塁打'}
EVENT_OF_CODE = {'SO': 'SO', 'OUT': 'OUT', 'SF': 'OUT', 'SAC': 'OUT',
                 'BB': 'BB', 'IBB': 'BB', 'HBP': 'BB', '1B': '1B', '2B': '2B', '3B': '3B', 'HR': 'HR'}
ON_BASE = np.array([e in ('BB', '1B', '2B', '3B', 'HR') for e in EVENTS])

# 走者のビット表現: 1塁=1, 2塁=2, 3塁=4
BASES_OF_RUNNER = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5, 6: 6, 7: 7}   # core.RUNNER_MAP の番号 → ビット
BASES_LABEL = {0: '走者なし', 1: '一塁', 2: '二塁', 4: '三塁', 3: '一二塁', 5: '一三塁', 6: '二三塁', 7: '満塁'}

C_RATE = 100      # 打者・投手の結果の割合をリーグ平均へ寄せる強さ（打席数）
K_TYPE = 50       # 相性の倍率を 1 へ寄せる強さ（期待件数）
MIN_TRANS = 10    # 経験分布を使う最低件数
EXTRA_INNINGS = 12
DEFAULT_SP_BF = 24   # 先発の記録がない投手の対戦打者数（2026年の先発の中央値）


# ============================================================
# モデルを作る（レギュラーシーズンの一球速報から）
# ============================================================
def build_sim_model(df_runs: pd.DataFrame, df_players: pd.DataFrame | None, type_model: dict) -> dict:
    """
    df_runs: calc_runs_after の出力（レギュラーシーズンのみ）
    type_model: core.build_type_model の結果（打者の長打型・投手の速球派の判定に使う）
    """
    d = attach_player_info(add_result_columns(df_runs), df_players)
    d['ev'] = d['結果'].map(EVENT_OF_CODE)
    d['bases'] = d['ランナー'].fillna('').map(RUNNER_MAP).fillna(0).astype(int).map(BASES_OF_RUNNER)
    d['side'] = np.where(d['イニング'].astype(str).str.contains('表'), 'away', 'home')
    # 打者・守備側の球団（短縮名）
    bat_raw = np.where(d['side'] == 'away', d['away_team_raw'], d['home_team_raw'])
    fld_raw = np.where(d['side'] == 'away', d['home_team_raw'], d['away_team_raw'])
    d['打者球団'] = pd.Series(bat_raw, index=d.index).apply(normalize_team_name)
    d['守備球団'] = pd.Series(fld_raw, index=d.index).apply(normalize_team_name)

    pa = d[d['ev'].notna()].copy()
    league = pa['ev'].value_counts(normalize=True).reindex(EVENTS).fillna(0).to_numpy()

    def _rates(col):
        cnt = pd.crosstab(pa[col], pa['ev']).reindex(columns=EVENTS, fill_value=0)
        n = cnt.sum(axis=1).to_numpy()[:, None]
        rates = (cnt.to_numpy() + C_RATE * league) / (n + C_RATE)
        return pd.DataFrame(rates, index=cnt.index, columns=EVENTS), cnt.sum(axis=1)

    bat_rates, bat_n = _rates('打者ID')
    pit_rates, pit_n = _rates('投手ID')

    # --- 相性（タイプ × タイプ × 結果）の倍率 ---
    b, p = type_model['batters'], type_model['pitchers']
    pa['bt'] = batter_type_label(pa['打者_打席'], pa['打者ID'].map(b['長打型']))
    pa['pt'] = pitcher_type_label(pa['投手_腕'], pa['投手ID'].map(p['速球派']))
    br = bat_rates.reindex(pa['打者ID']).to_numpy()
    pr = pit_rates.reindex(pa['投手ID']).to_numpy()
    raw = br * pr / league
    expv = raw / raw.sum(axis=1, keepdims=True)
    obs = pd.get_dummies(pa['ev']).reindex(columns=EVENTS, fill_value=0).to_numpy().astype(float)
    keep = pa['bt'].notna().to_numpy() & pa['pt'].notna().to_numpy() & ~np.isnan(expv).any(axis=1)
    cells = pd.DataFrame({'bt': pa['bt'].to_numpy()[keep], 'pt': pa['pt'].to_numpy()[keep]})
    type_mult = {}
    for (bt, pt), idx in cells.groupby(['bt', 'pt']).groups.items():
        o = obs[keep][idx].sum(axis=0)
        e = expv[keep][idx].sum(axis=0)
        type_mult[(bt, pt)] = (o + K_TYPE) / (e + K_TYPE)

    # --- 走者・アウトの進み方（経験分布）---
    grp = d.groupby(['game_id', 'イニング'], sort=False)
    d['next_out'] = grp['アウト'].shift(-1)
    d['next_bases'] = grp['bases'].shift(-1)
    d['next_runs'] = grp['runs_after'].shift(-1)
    is_last = d['next_out'].isna()
    ended = d['本文'].fillna('').str.contains('3アウト')
    d.loc[is_last, 'next_out'] = 3
    d.loc[is_last, 'next_bases'] = 0
    d.loc[is_last, 'next_runs'] = 0
    d['runs_play'] = d['runs_after'] - d['next_runs']
    t = d[d['ev'].notna() & (~is_last | ended)]
    trans = {}
    for (o, bs, ev), g in t.groupby(['アウト', 'bases', 'ev']):
        c = g.groupby(['next_out', 'next_bases', 'runs_play']).size()
        if c.sum() >= MIN_TRANS:
            keys = np.array([[int(a), int(bb), int(r)] for a, bb, r in c.index])
            prob = c.to_numpy() / c.sum()
            trans[(int(o), int(bs), ev)] = (keys, np.cumsum(prob))

    # --- 先発の対戦打者数・救援投手 ---
    d['投手ID'] = d['投手ID'].apply(normalize_player_id)
    first_p = d.groupby(['game_id', 'side'])['投手ID'].transform('first')
    d['先発'] = d['投手ID'] == first_p
    bf = d[d['先発']].groupby(['投手ID', 'game_id']).size()
    sp_bf = {pid: g.to_numpy() for pid, g in bf.groupby(level=0)}
    rel = d[~d['先発']].groupby(['守備球団', '投手ID']).size()
    bullpen = {team: g.droplevel(0).to_dict() for team, g in rel.groupby(level=0)}
    starters = d[d['先発']].groupby(['守備球団', '投手ID']).size()
    team_starters = {team: g.droplevel(0).sort_values(ascending=False).to_dict()
                     for team, g in starters.groupby(level=0)}

    # --- 名前・投打 ---
    names = {}
    if df_players is not None and not df_players.empty:
        pl = df_players.copy()
        pl['選手ID'] = pl['選手ID'].apply(normalize_player_id)
        names = dict(zip(pl['選手ID'], pl['選手名']))
        hands = pl.drop_duplicates('選手ID').set_index('選手ID')[['投', '打']]
    else:
        hands = pd.DataFrame(columns=['投', '打'])
    for col, ncol in (('打者ID', '選手名'), ('投手ID', '投手名')):
        for pid, nm in d.groupby(col)[ncol].first().items():
            names.setdefault(normalize_player_id(pid), nm)

    # 高速化のため、打席ごとの計算で使う値は dict に入れておく
    lookup = {
        'bat': {k: v for k, v in zip(bat_rates.index, bat_rates.to_numpy())},
        'pit': {k: v for k, v in zip(pit_rates.index, pit_rates.to_numpy())},
        'bats': hands['打'].dropna().to_dict(), 'arm': hands['投'].dropna().to_dict(),
        'power': b['長打型'].to_dict(), 'fast': p['速球派'].to_dict(),
    }
    return {'lookup': lookup, 'league': league, 'bat_rates': bat_rates, 'bat_n': bat_n, 'pit_rates': pit_rates, 'pit_n': pit_n,
            'type_mult': type_mult, 'trans': trans, 'sp_bf': sp_bf, 'bullpen': bullpen,
            'team_starters': team_starters, 'names': names, 'hands': hands, 'type_model': type_model,
            'pbp': d[['game_id', 'イニング', 'side', '打順', '打者ID', '投手ID', '先発', '打者球団', '守備球団']]}


# ============================================================
# 1打席の結果の確率
# ============================================================
def pa_probs(sim: dict, batter_id, pitcher_id, batter_form: str = '普通', pitcher_form: str = '普通') -> np.ndarray:
    """打者 × 投手の1打席の結果の確率（EVENTS の順）"""
    lg = sim['league']
    lk = sim['lookup']
    bid, pid = normalize_player_id(batter_id), normalize_player_id(pitcher_id)
    br = lk['bat'].get(bid, lg)
    pr = lk['pit'].get(pid, lg)
    raw = br * pr / lg
    prob = raw / raw.sum()

    # 相性: 打者の打席（両打は投手の逆側）× 長打型、投手の腕 × 速球派
    bats, arm = lk['bats'].get(bid), lk['arm'].get(pid)
    stance = {'右': '左', '左': '右'}.get(arm) if bats == '両' else bats
    power, fast = lk['power'].get(bid), lk['fast'].get(pid)
    if stance in ('右', '左') and arm in ('右', '左') and power is not None and fast is not None:
        bt = f"{stance}打・{'長打型' if power else '巧打型'}"
        pt = f"{arm}投・{'速球派' if fast else '技巧派'}"
        mult = sim['type_mult'].get((bt, pt))
        if mult is not None:
            prob = prob * mult
            prob = prob / prob.sum()

    # 調子: 出塁する結果の確率を打者は ×(1 ± FORM_RATE)、投手は逆向きに掛け、アウトの確率で合計を 1 に戻す
    f = (1 + FORM_RATE * FORM_SIGN.get(batter_form, 0)) * (1 - FORM_RATE * FORM_SIGN.get(pitcher_form, 0))
    if f != 1:
        on = min(prob[ON_BASE].sum() * f, 0.95)
        prob = np.where(ON_BASE, prob / prob[ON_BASE].sum() * on, prob / prob[~ON_BASE].sum() * (1 - on))
    return prob


def bullpen_probs(sim: dict, team: str, batter_id, batter_form: str = '普通') -> np.ndarray:
    """チームの救援投手全体と対戦するときの確率（救援での対戦打者数で重み付け）"""
    bp = sim['bullpen'].get(team, {})
    if not bp:
        return pa_probs(sim, batter_id, None, batter_form)
    tot = sum(bp.values())
    return sum(w / tot * pa_probs(sim, batter_id, pid, batter_form) for pid, w in bp.items())


# ============================================================
# 走者の進塁（経験分布が少ないときの単純なルール）
# ============================================================
def _simple_advance(outs: int, bases: int, ev: str) -> tuple[int, int, int]:
    r1, r2, r3 = bases & 1, (bases >> 1) & 1, (bases >> 2) & 1
    if ev in ('SO', 'OUT'):
        return outs + 1, bases, 0
    if ev == 'BB':
        if r1 and r2 and r3:
            return outs, 7, 1
        if r1 and r2:
            return outs, 7, 0
        if r1:
            return outs, bases | 2, 0
        return outs, bases | 1, 0
    if ev == '1B':
        return outs, 1 | (2 if r1 else 0) | (4 if r2 else 0), r3
    if ev == '2B':
        return outs, 2 | (4 if r1 else 0), r2 + r3
    if ev == '3B':
        return outs, 4, r1 + r2 + r3
    return outs, 0, 1 + r1 + r2 + r3   # HR


def _advance(sim: dict, rng, outs: int, bases: int, ev: str) -> tuple[int, int, int]:
    t = sim['trans'].get((outs, bases, ev))
    if t is None:
        return _simple_advance(outs, bases, ev)
    keys, cum = t
    i = int(np.searchsorted(cum, rng.random() * cum[-1], side='right'))
    no, nb, r = keys[min(i, len(keys) - 1)]
    return int(no), int(nb), int(r)


# ============================================================
# 試合のシミュレーション
# ============================================================
def prepare_game(sim: dict, away: dict, home: dict) -> dict:
    """
    away / home: {'team': 短縮名, 'lineup': [打者ID × 9], 'sp': 先発投手ID,
                  'batter_forms': [調子 × 9]（省略可）, 'sp_form': 調子（省略可）}
    相手の先発・救援と対戦するときの確率を前もって計算しておく
    """
    g = {}
    for side, me, opp in (('away', away, home), ('home', home, away)):
        forms = me.get('batter_forms') or ['普通'] * 9
        g[side] = {
            'lineup': [normalize_player_id(x) for x in me['lineup']],
            'vs_sp': np.array([pa_probs(sim, b, opp['sp'], f, opp.get('sp_form', '普通'))
                               for b, f in zip(me['lineup'], forms)]),
            'vs_bp': np.array([bullpen_probs(sim, opp['team'], b, f) for b, f in zip(me['lineup'], forms)]),
            'opp_sp': normalize_player_id(opp['sp']),
            'team': me['team'],
        }
    return g


def _sp_limit(sim: dict, rng, sp_id) -> int:
    hist = sim['sp_bf'].get(sp_id)
    if hist is None or len(hist) == 0:
        return DEFAULT_SP_BF
    return int(hist[rng.integers(len(hist))])


def simulate_game(sim: dict, game: dict, rng, record: bool = False) -> dict:
    """1試合をシミュレーションする。record=True なら打席ごとの経過も返す"""
    score = {'away': 0, 'home': 0}
    line = {'away': [], 'home': []}
    order = {'away': 0, 'home': 0}
    faced = {'away': 0, 'home': 0}            # 相手の先発がこれまでに対戦した打者数（攻撃側で数える）
    limit = {'away': _sp_limit(sim, rng, game['away']['opp_sp']),
             'home': _sp_limit(sim, rng, game['home']['opp_sp'])}
    log = []
    inning = 0
    while True:
        inning += 1
        for side in ('away', 'home'):
            if side == 'home' and inning >= 9 and score['home'] > score['away']:
                line['home'].append(None)   # 9回裏以降、ホームが勝っていれば攻撃しない
                break
            gs = game[side]
            outs, bases, runs = 0, 0, 0
            while outs < 3:
                i = order[side] % 9
                use_sp = faced[side] < limit[side]
                prob = gs['vs_sp'][i] if use_sp else gs['vs_bp'][i]
                ev = EVENTS[int(np.searchsorted(np.cumsum(prob), rng.random() * prob.sum(), side='right'))
                            if prob.sum() > 0 else 1]
                no, nb, r = _advance(sim, rng, outs, bases, ev)
                if record:
                    log.append({'回': inning, '表裏': '表' if side == 'away' else '裏', '打順': i + 1,
                                '打者': sim['names'].get(gs['lineup'][i], gs['lineup'][i]),
                                '投手': sim['names'].get(gs['opp_sp'], '先発') if use_sp else '救援陣',
                                'アウト': outs, '走者': BASES_LABEL.get(bases, ''), '結果': EVENT_LABEL[ev],
                                '得点': r, 'スコア': None})
                runs += r
                score[side] += r
                outs, bases = min(no, 3), nb
                order[side] += 1
                faced[side] += 1
                if record:
                    log[-1]['スコア'] = f"{score['away']}-{score['home']}"
                # サヨナラ: 9回裏以降にホームがリードしたら終了
                if side == 'home' and inning >= 9 and score['home'] > score['away']:
                    break
            line[side].append(runs)
        if inning >= 9 and score['home'] != score['away']:
            break
        if inning >= EXTRA_INNINGS:
            break
    res = 'away' if score['away'] > score['home'] else 'home' if score['home'] > score['away'] else 'tie'
    out = {'away': score['away'], 'home': score['home'], 'winner': res, 'innings': inning, 'line': line}
    if record:
        out['log'] = log
    return out


def simulate_many(sim: dict, away: dict, home: dict, n: int = 1000, seed: int | None = 0) -> dict:
    """n 試合をシミュレーションして集計する。1試合分の経過（最初の試合）も返す"""
    rng = np.random.default_rng(seed)
    game = prepare_game(sim, away, home)
    results = [simulate_game(sim, game, rng, record=(i == 0)) for i in range(n)]
    df = pd.DataFrame([{k: r[k] for k in ('away', 'home', 'winner', 'innings')} for r in results])
    inn = {}
    for side in ('away', 'home'):
        mat = np.full((n, EXTRA_INNINGS), np.nan)
        for j, r in enumerate(results):
            for k, v in enumerate(r['line'][side]):
                if v is not None:
                    mat[j, k] = v
        inn[side] = np.nanmean(mat[:, :9], axis=0)
    return {
        'n': n,
        '勝率': {'away': float((df['winner'] == 'away').mean()), 'home': float((df['winner'] == 'home').mean()),
                 '引き分け': float((df['winner'] == 'tie').mean())},
        '平均得点': {'away': float(df['away'].mean()), 'home': float(df['home'].mean())},
        '得点分布': {s: df[s].value_counts(normalize=True).sort_index() for s in ('away', 'home')},
        'スコア': (df.groupby(['away', 'home']).size().sort_values(ascending=False) / n).head(10),
        'イニング平均': inn,
        'サンプル': results[0],
        'results': df,
    }


# ============================================================
# 既定のスタメン・先発（アプリの初期値用）
# ============================================================
def last_lineup(sim: dict, team: str) -> list[str]:
    """そのチームの直近の試合の打順1〜9番（最初に打席に立った選手）"""
    d = sim['pbp']
    d = d[d['打者球団'] == team]
    if d.empty:
        return []
    gid = d['game_id'].astype(int)
    last = d[gid == gid.max()]
    first = last.dropna(subset=['打順']).drop_duplicates('打順').sort_values('打順')
    return [normalize_player_id(x) for x in first['打者ID'].head(9)]


def game_setup(df_game: pd.DataFrame) -> dict:
    """
    1試合分の打席データ（calc_runs_after の出力、打者ID・投手ID・打順つき）から、
    両チームの実際のスタメン（打順1〜9番に最初に入った選手）と先発投手を取り出す（検証用）
    """
    d = df_game.copy()
    d['side'] = np.where(d['イニング'].astype(str).str.contains('表'), 'away', 'home')
    out = {}
    for side, team_col in (('away', 'away_team_raw'), ('home', 'home_team_raw')):
        bat = d[d['side'] == side].dropna(subset=['打順']).drop_duplicates('打順').sort_values('打順')
        fld = d[d['side'] != side]
        out[side] = {'team': normalize_team_name(d[team_col].iloc[0]),
                     'lineup': [normalize_player_id(x) for x in bat['打者ID'].head(9)],
                     'sp': normalize_player_id(fld['投手ID'].iloc[0]) if len(fld) else None}
    return out


def team_batters(sim: dict, team: str) -> list[str]:
    """そのチームで打席に立った選手（打席の多い順）"""
    d = sim['pbp']
    c = d[d['打者球団'] == team]['打者ID'].apply(normalize_player_id).value_counts()
    return list(c.index)
