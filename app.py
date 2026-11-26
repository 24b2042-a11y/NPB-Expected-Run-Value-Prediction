"""
app.py — NPB 得点期待値予測アプリ

構成:
  1. メインページ … 得点期待値予測 + 選手個人成績
  2. 選手成績ページ … 打者・投手の成績を選んで閲覧
  3. 設定ページ   … データ取得・更新
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
import plotly.express as px
import importlib

# Streamlit Cloud は、GitHub に push された新しい app.py を実行しても、前回読み込んだ
# 古い core.py などをメモリ（sys.modules）に残したままにすることがある。その状態で
# 新しく追加した名前を import すると ImportError になるため、自作モジュールは毎回
# 読み込み直す。依存する順（scraper → core → game_sim → github_sync → player_scraper）に
# 1つずつ「読み込み直してから次を import する」こと。まとめて import してから reload すると、
# game_sim の import 時点ではまだ古い core が残っていて ImportError になる（2026-11-26）。
import sys
_LOCAL_MODULES = []
for _name in ('scraper', 'core', 'game_sim', 'github_sync', 'player_scraper'):
    _LOCAL_MODULES.append(importlib.reload(sys.modules[_name]) if _name in sys.modules
                          else importlib.import_module(_name))

# st.cache_data のキャッシュは「キャッシュする関数のコード」と引数で判定されるため、
# core.py だけを直すと古い core.py で作った結果（列の足りない df_woba など）が使われ続ける。
# core.py などの中身のハッシュを引数として渡し、自作モジュールが変わったらキャッシュを作り直す。
import hashlib
CODE_VERSION = hashlib.md5(b''.join(
    open(_m.__file__, 'rb').read() for _m in _LOCAL_MODULES
)).hexdigest()[:12]

from core import (
    build_model_from_dfs, predict_type_one, predict_type_all, find_batter_id, team_pitcher_weights,
    get_player_career, get_player_current_team, get_team_avg_runs,
    normalize_name, normalize_team_name, build_batter_woba, build_season_batting,
    RUNNER_LABEL, RUNNER_MAP, FORM_LEVELS, FORM_RATE, form_of,
)
import game_sim
# build_model_from_dfs の戻り値に df_team_batting_stats（対戦球団ごとの
# 打率・得点期待値）と df_situational_stats（状況別の打率・得点期待値）が
# 追加されたのに合わせて main() 側の unpack と page_main() の引数を拡張。
#
# 実測データ（df_team_batting_stats）を選手名で突き合わせる際は、必ず
# core.normalize_name() で正規化してから比較する。単純な文字列一致だと、
# 表記ゆれ（全角/半角、空白、中黒など）で「対戦データ（nf3）はあるのに
# 実測データが表示されない」という不具合が起きるため。
from scraper import scrape_games
from github_sync import (
    commit_game_files,
    load_details_from_github,
    get_existing_game_ids,
    save_batter_csv_to_github,
    load_batter_csv_from_github,
    save_pitcher_csv_to_github,
    load_pitcher_csv_from_github,
    save_players_csv_to_github,
    load_players_csv_from_github,
    get_legacy_game_ids,
)
from batter_scraper import scrape_all_batters, TEAMS as BATTER_TEAMS
from pitcher_scraper import scrape_all_pitchers
from player_scraper import scrape_players, collect_player_ids, merge_players

st.set_page_config(
    page_title='NPB 得点期待値予測',
    page_icon='⚾',
    layout='wide',
    initial_sidebar_state='expanded',
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BATTER_CSV = os.path.join(BASE_DIR, 'data', 'all_batters_situational.csv')
PITCHER_CSV = os.path.join(BASE_DIR, 'data', 'all_pitchers.csv')
PLAYERS_CSV = os.path.join(BASE_DIR, 'data', 'players.csv')
GAMEDATA_DIR = os.path.join(BASE_DIR, 'data', 'gamedata')
STATS_DIR = os.path.join(BASE_DIR, 'data', '2023~2025打撃データ')
START_ID   = 2021038624
CURRENT_SEASON = 2026   # 「今シーズン」として表示する年（nf3 の y=0 = 当年度のデータ）

RUNNER_STR_MAP = {
    (False, False, False): '走者なし',
    (True,  False, False): '1塁',
    (False, True,  False): '2塁',
    (False, False, True):  '3塁',
    (True,  True,  False): '1,2塁',
    (True,  False, True):  '1,3塁',
    (False, True,  True):  '2,3塁',
    (True,  True,  True):  '満塁',
}


# ============================================================
# キャッシュ
# ============================================================
def load_details_local(data_dir: str = GAMEDATA_DIR) -> list[pd.DataFrame]:
    """
    data/gamedata/*_details.csv をローカルから読み込む（load_details_from_github のローカル版）。
    GitHub トークン未設定時に使う。
    """
    dfs = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*_details.csv'))):
        fname = os.path.basename(path)
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            continue
        df['game_id'] = fname.split('_', 1)[0]
        card = fname.split('_', 1)[1].replace('_details.csv', '')
        home_raw, away_raw = card.split('vs.', 1) if 'vs.' in card else (None, None)
        df['home_team_raw'] = home_raw
        df['away_team_raw'] = away_raw
        dfs.append(df)
    return dfs


@st.cache_data(show_spinner='データを読み込んでいます...')
def load_model(token: str | None, repo_name: str | None, branch: str | None, code_version: str = CODE_VERSION):
    # token が無い（None）ときは GitHub を使わず、リポジトリ同梱のローカル data/ を読む
    if token:
        dfs, _ = load_details_from_github(token, repo_name, branch)
        batter_df = load_batter_csv_from_github(token, repo_name, branch)
        df_players = load_players_csv_from_github(token, repo_name, branch)
    else:
        dfs = load_details_local()
        batter_df = None
        df_players = None
    # 選手マスタ（投打・球速）。打者・投手のタイプ分けに使う
    if df_players is None and os.path.exists(PLAYERS_CSV):
        df_players = pd.read_csv(PLAYERS_CSV, encoding='utf-8-sig', dtype={'選手ID': str, '背番号': str})

    # 対戦成績CSVはまずGitHub上の最新版を試し、なければリポジトリ同梱のローカル版を使う
    if batter_df is None:
        batter_df = pd.read_csv(BATTER_CSV, encoding='utf-8-sig')

    return build_model_from_dfs(dfs, batter_df, STATS_DIR, df_players)


def reload_model():
    load_model.clear()
    load_season_batting.clear()
    st.rerun()


@st.cache_data(show_spinner='投手データを読み込んでいます...')
def load_pitchers(token: str | None, repo_name: str | None, branch: str | None, code_version: str = CODE_VERSION):
    """投手成績CSV（data/all_pitchers.csv）。まだ無い場合は None"""
    df = load_pitcher_csv_from_github(token, repo_name, branch) if token else None
    if df is None and os.path.exists(PITCHER_CSV):
        df = pd.read_csv(PITCHER_CSV, encoding='utf-8-sig')
    return df


@st.cache_data(show_spinner='今シーズンの成績を集計しています...')
def load_season_batting(token: str | None, repo_name: str | None, branch: str | None, code_version: str = CODE_VERSION):
    """対戦相手別の打者成績CSVを選手×球団ごとに合計した、今シーズンの通算成績"""
    batter_df = load_batter_csv_from_github(token, repo_name, branch) if token else None
    if batter_df is None:
        batter_df = pd.read_csv(BATTER_CSV, encoding='utf-8-sig')
    return build_season_batting(batter_df)


def get_github_cfg() -> dict | None:
    try:
        cfg = st.secrets['github']
        token = str(cfg['token']).strip()
        # 未入力やプレースホルダーのままだと HTTP ヘッダーに載せられず
        # UnicodeEncodeError になるため、未設定として扱う
        if not token or not token.isascii():
            return None
        return {
            'token':     token,
            'repo_name': cfg['repo_name'],
            'branch':    cfg.get('branch', 'main'),
        }
    except Exception:
        return None


# ============================================================
# 可視化
# ============================================================
def plot_re24_heatmap(re24, highlight_out=None, highlight_runner=None):
    z    = re24.values.astype(float)
    text = np.where(np.isnan(z), 'N/A', np.round(z, 2).astype(str))
    shapes = []
    if highlight_out is not None and highlight_runner is not None:
        r_idx = RUNNER_MAP.get(highlight_runner, 0)
        shapes.append(dict(
            type='rect',
            x0=r_idx - 0.5, x1=r_idx + 0.5,
            y0=highlight_out - 0.5, y1=highlight_out + 0.5,
            line=dict(color='crimson', width=3),
        ))
    fig = go.Figure(go.Heatmap(
        z=z, x=RUNNER_LABEL,
        y=['0アウト', '1アウト', '2アウト'],
        text=text, texttemplate='%{text}',
        colorscale='RdYlGn', showscale=True,
        colorbar=dict(title='期待得点'),
        zmin=0, zmax=float(np.nanmax(z)) if not np.all(np.isnan(z)) else 1,
    ))
    fig.update_layout(
        title='RE24 得点期待値行列', xaxis_title='ランナー状態',
        yaxis_title='アウトカウント', shapes=shapes,
        height=300, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def plot_24_bar(df_all, batter_name, opponent):
    df   = df_all.copy()
    df['場面'] = df['アウト'].astype(str) + 'out / ' + df['ランナー']
    df_p = df[df['補正後期待得点'].notna()]
    fig  = px.bar(
        df_p, x='場面', y='補正後期待得点', color='アウト',
        color_continuous_scale='Blues',
        text=df_p['補正後期待得点'].round(3),
        title=f'{batter_name} vs {opponent} — 全 24 場面',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45, height=420,
        margin=dict(l=20, r=20, t=50, b=130),
        showlegend=False, coloraxis_showscale=False,
    )
    return fig


def plot_career_trend(df_career):
    """過去成績の推移（打率・出塁率・長打率・OPS）"""
    fig = go.Figure()
    for col, color in [('打率', '#1f77b4'), ('出塁率', '#ff7f0e'),
                       ('長打率', '#2ca02c'), ('OPS', '#d62728')]:
        fig.add_trace(go.Scatter(
            x=df_career['年度'], y=df_career[col],
            mode='lines+markers', name=col, line=dict(color=color),
        ))
    fig.update_layout(
        title='過去成績の推移', xaxis_title='年度', yaxis_title='値',
        xaxis=dict(dtick=1), height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation='h', y=1.15),
    )
    return fig


# ============================================================
# 実測データ（df_team_batting_stats）を選手名で絞り込むヘルパー
#   単純な文字列一致だと、全角/半角・空白・中黒などの表記ゆれで
#   「対戦データ（nf3）はあるのに実測データが出ない」不具合が起きるため、
#   必ず normalize_name() で正規化してから比較する。
# ============================================================
def filter_team_batting_stats_by_batter(df_team_batting_stats: pd.DataFrame, batter: str) -> pd.DataFrame:
    if df_team_batting_stats.empty:
        return df_team_batting_stats
    batter_key = normalize_name(batter)
    name_keys  = df_team_batting_stats['選手名'].apply(normalize_name)
    return df_team_batting_stats[name_keys == batter_key]


# ============================================================
# 投手の成績（data/all_pitchers.csv）
# ============================================================
def _ip_str(ip) -> str:
    """投球回の実数（68.333）を野球の表記（68.1）に戻す"""
    if ip is None or pd.isna(ip):
        return 'N/A'
    whole = int(ip + 1e-9)
    outs  = int(round((ip - whole) * 3))
    if outs == 3:
        whole, outs = whole + 1, 0
    return f'{whole}.{outs}'


def _fmt(v, f: str = '{:.2f}') -> str:
    return 'N/A' if v is None or pd.isna(v) else f.format(v)


def _int_str(v) -> str:
    return 'N/A' if v is None or pd.isna(v) else str(int(v))


def page_pitcher_section(df_pitchers, opponent: str | None = None):
    st.divider()
    st.subheader('🧢 投手の成績')
    render_pitcher_stats(df_pitchers, opponent, key_prefix='main_pitcher')


def render_pitcher_stats(df_pitchers, opponent: str | None = None, key_prefix: str = 'pitcher'):
    if df_pitchers is None or df_pitchers.empty:
        st.caption('投手データがありません。「⚙️ 設定」ページの「投手成績データを更新」で取得してください。')
        return

    tot = df_pitchers[df_pitchers['区分種別'] == '通算'].copy()
    if tot.empty:
        st.caption('投手の通算成績がありません。投手データを更新してください。')
        return

    tot['ラベル'] = tot['選手名'] + '（' + tot['球団'].astype(str) + '）'
    tot = tot.sort_values(['球団', '選手名'])

    # 打者の対戦相手（球団）の投手に絞る。打者はその球団の投手と対戦するため
    only_opp = False
    if opponent:
        only_opp = st.checkbox(
            f'対戦相手（{opponent}）の投手に絞る', value=True, key=f'{key_prefix}_only_opp',
        )
    cand = tot
    if only_opp:
        opp_norm = normalize_team_name(opponent)
        narrowed = tot[tot['球団'].apply(normalize_team_name) == opp_norm]
        if narrowed.empty:
            st.caption(f'{opponent} の投手データが見つかりません。全球団から選べます。')
        else:
            cand = narrowed

    query = st.text_input('🔍 投手名を入力', placeholder='例: 村上、戸郷', key=f'{key_prefix}_query')
    if query:
        cand = cand[cand['選手名'].str.contains(query, regex=False)]
    if cand.empty:
        st.warning(f'「{query}」に一致する投手が見つかりません。')
        return

    label = st.selectbox('投手を選択', cand['ラベル'].tolist(), key=f'{key_prefix}_select')
    sel   = cand[cand['ラベル'] == label].iloc[0]
    mine  = df_pitchers[(df_pitchers['選手名_key'] == sel['選手名_key'])
                        & (df_pitchers['TM'] == sel['TM'])]

    arm = sel.get('腕')
    arm_txt = f'　{arm}投げ' if isinstance(arm, str) and arm else ''
    st.markdown(f"**{sel['選手名']}**　{sel['球団']}{arm_txt}")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric('防御率', _fmt(sel['防御率']))
    a2.metric('投球回', _ip_str(sel['投球回']))
    a3.metric('登板（先発/救援）',
              f"{_int_str(sel['試合'])}（{_int_str(sel['先発'])}/{_int_str(sel['救援'])}）")
    a4.metric('被打者', _int_str(sel['被打者']))

    b1, b2, b3, b4 = st.columns(4)
    b1.metric('K%',  _fmt(sel['K%'] * 100, '{:.1f}%'))
    b2.metric('BB%', _fmt(sel['BB%'] * 100, '{:.1f}%'))
    b3.metric('WHIP', _fmt(sel['WHIP']))
    b4.metric('FIP',  _fmt(sel['FIP']))
    st.caption('FIP は全投手の通算から求めた定数で補正しており、防御率と同じ目盛りで比べられます。')

    show_cols = ['区分名', '試合', '投球回', '被安', '被本', '三振', '四球', '死球',
                 '自責', '防御率', 'K%', 'BB%', 'WHIP', 'FIP']

    def split_table(kind: str) -> pd.DataFrame:
        d = mine[mine['区分種別'] == kind].sort_values('投球回', ascending=False).copy()
        d['投球回'] = d['投球回'].apply(_ip_str)
        for c in ('K%', 'BB%'):
            d[c] = (d[c] * 100).round(1)
        for c in ('防御率', 'WHIP', 'FIP'):
            d[c] = d[c].round(2)
        return d[[c for c in show_cols if c in d.columns]].reset_index(drop=True)

    tab_opp, tab_std = st.tabs(['対戦相手別', '球場別'])
    with tab_opp:
        d = split_table('対戦相手')
        if d.empty:
            st.caption('対戦相手別のデータがありません（個別ページを取得できていない可能性があります）。')
        else:
            st.dataframe(d, width='stretch')
    with tab_std:
        d = split_table('球場')
        if d.empty:
            st.caption('球場別のデータがありません（個別ページを取得できていない可能性があります）。')
        else:
            st.dataframe(d, width='stretch')
    st.caption('※ 対戦相手別・球場別の K%・BB% は、被打者を「3×投球回＋被安＋四球＋死球」で'
               '推定した概算です。投球回が少ないと数値が大きくぶれます。')


# ============================================================
# 打者の個人成績（メインページ・選手成績ページ共通）
# ============================================================
def render_batter_stats(batter, df_woba, df_career, df_team_batting_stats, df_season=None):
    tab_names = ['今シーズン（対戦相手別 wOBA）', '対戦球団別（実測打率・得点期待値）', '過去3年成績']
    if df_season is not None:
        tab_names = [f'{CURRENT_SEASON}年成績'] + tab_names
    tabs = st.tabs(tab_names)
    if df_season is not None:
        with tabs[0]:
            render_season_batting(batter, df_season)
        tab_recent, tab_real, tab_career = tabs[1:]
    else:
        tab_recent, tab_real, tab_career = tabs

    with tab_recent:
        df_b = (df_woba[df_woba['選手名'] == batter][['区分名','打席','wOBA']]
                .query('打席 >= 10').sort_values('wOBA', ascending=False))
        df_b['wOBA'] = df_b['wOBA'].round(3)
        df_b.columns = ['対戦相手','打席数','wOBA']
        if df_b.empty:
            st.caption('対戦相手別データ（打席10以上）がありません。')
        else:
            st.dataframe(df_b, width='stretch')

    with tab_real:
        df_r = (filter_team_batting_stats_by_batter(df_team_batting_stats, batter)
                [['対戦球団','打席','打数','安打','打率','得点期待値']]
                .sort_values('得点期待値', ascending=False))
        if df_r.empty:
            st.caption('実測データ（プレーバイプレー）がありません。')
        else:
            st.dataframe(df_r, width='stretch')
            st.caption('打率・得点期待値は標本数が少ない（打席5未満）場合 N/A になります。')

    with tab_career:
        career = get_player_career(df_career, batter)
        if career.empty:
            st.caption('過去成績データが見つかりません。')
        else:
            df_show = career[['年度','所属球団','試合','打席','打率','出塁率','長打率','OPS',
                              '本塁打','打点','盗塁']].reset_index(drop=True)
            st.dataframe(df_show, width='stretch')
            if len(career) >= 2:
                st.plotly_chart(plot_career_trend(career), width='stretch')


# ============================================================
# 今シーズンの成績（個人・一覧）
# ============================================================
BATTER_LB_COLS  = ['選手名', '球団', '試合', '打席', '打数', '安打', '本塁', '打点', '盗塁',
                   '四球', '三振', '打率', '出塁率', '長打率', 'OPS', 'wOBA']
PITCHER_LB_COLS = ['選手名', '球団', '腕', '試合', '先発', '救援', '投球回', '被打者',
                   '防御率', 'K%', 'BB%', 'WHIP', 'FIP']
LOWER_IS_BETTER = {'防御率', 'FIP', 'WHIP', 'BB%'}


def render_season_batting(batter, df_season):
    """選んだ打者の今シーズン通算（対戦相手別成績の合計）"""
    key  = normalize_name(batter)
    mine = df_season[df_season['選手名_key'] == key].sort_values('打席', ascending=False)
    if mine.empty:
        st.caption(f'{batter} の{CURRENT_SEASON}年成績データがありません。')
        return

    r = mine.iloc[0]

    def rate(name):
        return 'N/A' if name not in r.index or pd.isna(r[name]) else f'{r[name]:.3f}'

    def cnt(name):
        return _int_str(r[name]) if name in r.index else 'N/A'

    a1, a2, a3, a4 = st.columns(4)
    a1.metric('打率', rate('打率'))
    a2.metric('出塁率', rate('出塁率'))
    a3.metric('長打率', rate('長打率'))
    a4.metric('OPS', rate('OPS'))
    b1, b2, b3, b4 = st.columns(4)
    b1.metric('wOBA', rate('wOBA'))
    b2.metric('本塁打', cnt('本塁'))
    b3.metric('打点', cnt('打点'))
    b4.metric('盗塁', cnt('盗塁'))

    cols = ['球団', '試合', '打席', '打数', '安打', '2B', '3B', '本塁', '打点', '盗塁',
            '四球', '死球', '三振', '打率', '出塁率', '長打率', 'OPS', 'wOBA']
    d = mine[[c for c in cols if c in mine.columns]].copy()
    for c in ('打率', '出塁率', '長打率', 'OPS', 'wOBA'):
        if c in d.columns:
            d[c] = d[c].round(3)
    st.dataframe(d.reset_index(drop=True), width='stretch')
    note = f'{CURRENT_SEASON}年の対戦相手別成績（nf3）を合計した通算です。'
    if len(mine) > 1:
        note += '移籍した選手は球団ごとに行が分かれ、上の指標は打席数の多い球団のものです。'
    st.caption(note)


def render_season_leaderboard(df_season, df_pitchers):
    """今シーズンの打者・投手の成績一覧（列の見出しをクリックして並べ替えも可能）"""
    sub_b, sub_p = st.tabs(['🏏 打者', '⚾ 投手'])

    with sub_b:
        if df_season is None or df_season.empty:
            st.caption('打者の成績データがありません。「⚙️ 設定」ページで打者データを更新してください。')
        else:
            teams = sorted(df_season['球団'].dropna().astype(str).unique().tolist())
            c1, c2, c3 = st.columns(3)
            team   = c1.selectbox('球団', ['全球団'] + teams, key='lb_b_team')
            min_pa = c2.number_input('最低打席数', min_value=0, max_value=800, value=100,
                                     step=10, key='lb_b_minpa')
            sort_opts = [c for c in ('OPS', 'wOBA', '打率', '出塁率', '長打率', '本塁', '打点', '盗塁', '打席')
                         if c in df_season.columns]
            sort_col = c3.selectbox('並べ替え', sort_opts, key='lb_b_sort')

            d = df_season[df_season['打席'] >= min_pa]
            if team != '全球団':
                d = d[d['球団'].astype(str) == team]
            if d.empty:
                st.caption('条件に合う打者がいません。')
            else:
                d = d.sort_values(sort_col, ascending=False, na_position='last')
                show = d[[c for c in BATTER_LB_COLS if c in d.columns]].copy()
                for c in ('打率', '出塁率', '長打率', 'OPS', 'wOBA'):
                    if c in show.columns:
                        show[c] = show[c].round(3)
                st.dataframe(show.reset_index(drop=True), width='stretch', height=600)
                st.download_button(
                    '📥 CSV ダウンロード',
                    data=show.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name=f'{CURRENT_SEASON}_打者成績.csv', mime='text/csv', key='lb_b_dl',
                )
            st.caption('対戦相手別成績の合計から計算しています。セ・リーグは投手の打席も含まれるため、'
                       '最低打席数で絞って見てください。')

    with sub_p:
        tot = None
        if df_pitchers is not None and not df_pitchers.empty:
            tot = df_pitchers[df_pitchers['区分種別'] == '通算']
        if tot is None or tot.empty:
            st.caption('投手の成績データがありません。「⚙️ 設定」ページで投手データを更新してください。')
        else:
            teams = sorted(tot['球団'].dropna().astype(str).unique().tolist())
            c1, c2, c3 = st.columns(3)
            team   = c1.selectbox('球団', ['全球団'] + teams, key='lb_p_team')
            min_ip = c2.number_input('最低投球回', min_value=0, max_value=250, value=30,
                                     step=5, key='lb_p_minip')
            sort_opts = [c for c in ('防御率', 'FIP', 'WHIP', 'K%', 'BB%', '投球回', '試合')
                         if c in tot.columns]
            sort_col = c3.selectbox('並べ替え', sort_opts, key='lb_p_sort')

            d = tot[tot['投球回'] >= min_ip]
            if team != '全球団':
                d = d[d['球団'].astype(str) == team]
            if d.empty:
                st.caption('条件に合う投手がいません。')
            else:
                d = d.sort_values(sort_col, ascending=(sort_col in LOWER_IS_BETTER), na_position='last')
                show = d[[c for c in PITCHER_LB_COLS if c in d.columns]].copy()
                show['投球回'] = show['投球回'].apply(_ip_str)
                for c in ('K%', 'BB%'):
                    if c in show.columns:
                        show[c] = (show[c] * 100).round(1)
                for c in ('防御率', 'WHIP', 'FIP'):
                    if c in show.columns:
                        show[c] = show[c].round(2)
                st.dataframe(show.reset_index(drop=True), width='stretch', height=600)
                st.download_button(
                    '📥 CSV ダウンロード',
                    data=show.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
                    file_name=f'{CURRENT_SEASON}_投手成績.csv', mime='text/csv', key='lb_p_dl',
                )
            st.caption('通算成績（リーグ戦＋交流戦）です。防御率・FIP・WHIP・BB% は小さいほど良いので、'
                       'それらで並べ替えると小さい順になります。')


# ============================================================
# 選手成績ページ: 打者・投手の成績を選んで閲覧する
# ============================================================
def page_players(df_woba, df_career, df_team_batting_stats, df_pitchers, df_season=None):
    st.title('👥 選手成績')
    st.caption('打者・投手を選んで成績を閲覧できます。')
    st.divider()

    tab_bat, tab_pit, tab_lb = st.tabs(['🏏 打者', '⚾ 投手', f'📋 {CURRENT_SEASON}年 成績一覧'])

    with tab_bat:
        names = df_woba[['選手名', '選手名_key']].drop_duplicates()

        # 所属球団（過去3年成績データの最新年度）で絞り込めるようにする
        team_of = {}
        if df_career is not None and not df_career.empty:
            latest = df_career.sort_values('年度').groupby('選手名_key').tail(1)
            team_of = dict(zip(latest['選手名_key'], latest['所属球団']))
        names = names.assign(所属=names['選手名_key'].map(team_of))

        col_t, col_q = st.columns(2)
        with col_t:
            teams = sorted(names['所属'].dropna().unique().tolist())
            team_sel = st.selectbox('球団（所属）', ['全球団'] + teams, key='players_team')
        with col_q:
            query = st.text_input('🔍 選手名を入力', placeholder='例: 田中、鈴木、村上',
                                  key='players_batter_query')

        pool = names if team_sel == '全球団' else names[names['所属'] == team_sel]
        candidates = sorted(pool['選手名'].unique().tolist())
        if query:
            candidates = [b for b in candidates if query in b]

        if not candidates:
            st.warning('条件に一致する選手が見つかりません。')
        else:
            batter = st.selectbox('選手を選択', candidates, key='players_batter_select')
            team = get_player_current_team(df_career, batter)
            st.markdown(f'### {batter}' + (f'　{team}' if team else ''))
            render_batter_stats(batter, df_woba, df_career, df_team_batting_stats, df_season)
        st.caption('所属球団は過去3年成績データの最新年度のものです。')

    with tab_pit:
        render_pitcher_stats(df_pitchers, None, key_prefix='players_pitcher')

    with tab_lb:
        render_season_leaderboard(df_season, df_pitchers)


# ============================================================
# メインページ: 得点期待値予測 + 選手個人成績
# ============================================================
def _form_caption(f: dict | None, role: str) -> str:
    """直近の成績からの調子の判定を1行で"""
    if not f:
        return '直近の成績: データなし'
    unit = '試合' if role == '打者' else '登板'
    label = 'wOBA' if role == '打者' else '被wOBA'
    rec = f"{f['直近wOBA']:.3f}" if pd.notna(f['直近wOBA']) else '—'
    return (f"直近 {int(f['直近試合数'])} {unit}（{int(f['直近打席'])} 打席）の{label} {rec}"
            f"／シーズン {f['シーズンwOBA']:.3f} → 判定: **{f['調子']}**（選んだときだけ使います）")


def page_main(re24, counts, df_woba, league_avg, n_games, n_pa, df_career, df_team_runs,
              df_team_batting_stats, df_situational_stats, df_pitchers=None, df_season=None,
              type_model=None):
    st.title('⚾ 得点期待値予測')
    if type_model is None:
        st.error('相性モデルを作れませんでした（打席データに打者ID・投手ID の列がありません）。'
                 '「⚙️ 設定」の「既存試合の再取得」を実行してください。')
        st.stop()
    lg, k = type_model['lg'], type_model['k']
    st.caption(f'使用データ: レギュラーシーズン {n_games} 試合 / {n_pa:,} 打席  |  リーグ平均 wOBA: {lg:.3f}')
    st.divider()

    # ---- 選手名テキスト入力（インクリメンタルサーチ） ----
    # 一球速報に打席のある打者だけ（nf3 にだけいる打者は投手などで、打席は合計でも20打席に満たない）
    tm_names = set(type_model['batters']['選手名'].apply(normalize_name)) if type_model is not None else None
    all_batters = sorted(n for n in df_woba['選手名'].unique().tolist()
                         if tm_names is None or normalize_name(n) in tm_names)
    query = st.text_input('🔍 選手名を入力', placeholder='例: 田中、鈴木、村上')

    if query:
        candidates = [b for b in all_batters if query in b]
    else:
        candidates = all_batters

    if not candidates:
        st.warning(f'「{query}」に一致する選手が見つかりません。')
        return

    batter = st.selectbox('選手を選択', candidates, key='batter_select')

    # ---- 選手プロフィール（所属球団 + 過去3年成績） ----
    team = get_player_current_team(df_career, batter)
    if team:
        st.markdown(f'**所属球団：{team}**')

    st.divider()

    # ---- 場面設定 ----
    st.subheader('場面設定')
    col_opp, col_out = st.columns(2)
    with col_opp:
        opponents = sorted(df_woba['区分名'].unique().tolist())
        opponent  = st.selectbox('対戦相手', opponents, key='opponent_select')
        # 対戦投手: 相手チームで学習データに登板した投手（対戦打者の多い順）
        team_w = team_pitcher_weights(type_model, opponent)
        tp = type_model['pitchers'].loc[list(team_w)].sort_values('対戦打者', ascending=False) if team_w else None
        ALL_P = '（指定しない：相手チームの投手全体）'
        p_options = [ALL_P] + (list(tp.index) if tp is not None else [])

        def _p_label(pid):
            if pid == ALL_P:
                return pid
            r = tp.loc[pid]
            arm = {'右': '右投', '左': '左投'}.get(r['投'], '?')
            return f"{r['選手名']}（{arm}・{'速球派' if r['速球派'] else '技巧派'}／対戦打者 {int(r['対戦打者'])}）"
        pitcher = st.selectbox('対戦投手', p_options, format_func=_p_label, key='pitcher_select')
    with col_out:
        out = st.radio('アウトカウント', [0, 1, 2],
                       format_func=lambda x: f'{x} アウト',
                       horizontal=True, key='out_radio')

    st.markdown('**ランナー状態**')
    rc1, rc2, rc3 = st.columns(3)
    r1 = rc1.checkbox('1塁', key='r1')
    r2 = rc2.checkbox('2塁', key='r2')
    r3 = rc3.checkbox('3塁', key='r3')
    runner = RUNNER_STR_MAP[(r1, r2, r3)]
    st.caption(f'選択中: **{runner}**')

    # ---- 調子（既定は普通。直近の成績からの判定は参考に表示） ----
    st.markdown('**調子**')
    batter_id = find_batter_id(type_model, batter)
    fc1, fc2 = st.columns(2)
    with fc1:
        batter_form = st.radio(f'打者（{batter}）', FORM_LEVELS, index=1, horizontal=True, key='batter_form')
        st.caption(_form_caption(form_of(type_model.get('form'), '打者', batter_id), '打者'))
    with fc2:
        if pitcher == ALL_P:
            pitcher_form = st.radio(f'{opponent} の投手陣', FORM_LEVELS, index=1, horizontal=True,
                                    key='pitcher_form')
            st.caption('投手を選ぶと、直近の登板からの判定を表示します。')
        else:
            pitcher_form = st.radio(f"投手（{tp.loc[pitcher, '選手名']}）", FORM_LEVELS, index=1,
                                    horizontal=True, key='pitcher_form')
            st.caption(_form_caption(form_of(type_model.get('form'), '投手', pitcher), '投手'))
    st.caption(f'好調・不調は打者の wOBA（投手は被 wOBA）を ±{FORM_RATE:.0%} 補正します。'
               '2026年のデータでは、直近の調子が次の試合の成績を当てる効果は確認できていません（参考機能）。')

    st.divider()

    # ---- 1場面の予測結果（打者タイプ × 投手タイプの相性モデル） ----
    weights = team_w if pitcher == ALL_P else {pitcher: 1.0}
    res = predict_type_one(re24, type_model, batter_id, weights, out, runner, batter_form, pitcher_form)
    if batter_id is None:
        st.warning(f'※ {batter} は一球速報に打席がないため、打者 wOBA はリーグ平均として計算しています。')

    st.subheader('📊 予測結果')
    m1, m2, m3 = st.columns(3)
    m1.metric('基礎 RE24', str(res['基礎RE24'] or 'N/A'))
    m2.metric('対戦 wOBA 予測', f"{res['対戦wOBA予測']:.3f}",
              help='打者 wOBA ＋ 投手補正 ＋ 相性補正')
    delta_str = ''
    if res['補正後期待得点'] is not None and res['基礎RE24'] is not None:
        delta_str = f"{res['補正後期待得点'] - res['基礎RE24']:+.3f}"
    m3.metric('補正後期待得点', str(res['補正後期待得点'] if res['補正後期待得点'] is not None else 'N/A'),
              delta=delta_str)

    t1, t2, t3 = st.columns(3)
    t1.metric('打者タイプ', res['打者タイプ'] or '不明')
    if pitcher == ALL_P:
        t2.metric('投手タイプ', f'{opponent} の投手全体', help='登板の多さで重み付けした平均')
    else:
        t2.metric('投手タイプ', res['投手タイプ'] or '不明')
    avg_runs, n_team_games = get_team_avg_runs(df_team_runs, opponent)
    t3.metric(f'{opponent} の1試合平均得点',
              f'{avg_runs:.2f}' if avg_runs is not None else 'N/A',
              help=f'集計試合数: {n_team_games} 試合' if n_team_games else 'データなし')

    # ---- 投手の成績（予測の「投手補正」の元） ----
    if pitcher != ALL_P:
        pr = tp.loc[pitcher]
        velo = f"{pr['速球']:.1f} km/h" if pd.notna(pr['速球']) else '—'
        st.caption(f"投手 {pr['選手名']}: 対戦打者 {int(pr['対戦打者'])}、被 wOBA {pr['被wOBA']:.3f}"
                   f"（リーグ平均へ収縮後 {pr['w']:.3f}）、奪三振率 {pr['K%']:.1%}、速球の平均 {velo}")

    # ---- 計算の内訳 ----
    st.caption(
        f"対戦 wOBA 予測 = 打者 wOBA {res['打者wOBA']:.3f} ＋ 投手補正 {res['投手補正']:+.3f} "
        f"＋ 相性補正 {res['相性補正']:+.3f} = **{res['対戦wOBA予測']:.3f}**　"
        f"（打者 wOBA は一球速報 {res['打者打席']} 打席をリーグ平均へ収縮）"
    )
    if res['基礎RE24'] is not None and res['補正後期待得点'] is not None:
        st.caption(
            f"補正後期待得点 = RE24 {res['基礎RE24']:.3f} ＋ {k:.2f} × "
            f"({res['対戦wOBA予測']:.3f} − リーグ平均 {lg:.3f}) = **{res['補正後期待得点']:.3f}**"
        )
    with st.expander('打者タイプ × 投手タイプの相性表'):
        st.caption('数値はタイプ同士の対戦で、打者・投手の実力から期待される wOBA との差（×1000）。'
                   'プラスなら打者有利。打席の少ないマスは 0 に近づけている（ベイズ収縮）。')
        mt = (type_model['matchup'] * 1000).round(1)
        st.dataframe(mt.style.background_gradient(cmap='RdYlGn', vmin=-30, vmax=30), width='stretch')
        st.caption('各マスの打席数')
        st.dataframe(type_model['matchup_n'], width='stretch')
        pr = type_model['params']
        st.caption(f"長打型 = ISO {pr['ISO境目']:.3f} 以上（ISO = 長打の余分な塁打 ÷ 打数）。"
                   '速球派 = 速球の平均球速と奪三振率の z 値の平均が 0 以上。'
                   '両打の打者は投手の逆側の打席として数える。')

    # ---- 対戦球団ごとの実測打率・得点期待値（状況別ではない） ----
    row = filter_team_batting_stats_by_batter(df_team_batting_stats, batter)
    row = row[row['対戦球団'] == opponent]
    if not row.empty:
        r = row.iloc[0]
        e1, e2, e3 = st.columns(3)
        e1.metric(f'実測打率 vs {opponent}',
                 f"{r['打率']:.3f}" if r['打率'] is not None else 'N/A')
        e2.metric(f'実測得点期待値 vs {opponent}',
                 f"{r['得点期待値']:.3f}" if r['得点期待値'] is not None else 'N/A')
        e3.metric('実測打席数（対戦球団）', f"{int(r['打席'])}")
        st.caption('※ 実際のプレーバイプレーから集計した数値（アウト・ランナー状況は区切らない）')
    else:
        st.caption(f'{batter} vs {opponent} の実測データがありません。')

    st.divider()

    # ---- RE24 ヒートマップ ----
    st.subheader('🗺 RE24 行列')
    hl_runner = runner if runner != '走者なし' else ''
    st.plotly_chart(plot_re24_heatmap(re24, out, hl_runner), width='stretch')
    with st.expander('各マスのサンプル数'):
        st.dataframe(
            pd.DataFrame(counts, index=['0アウト','1アウト','2アウト'], columns=RUNNER_LABEL),
            width='stretch',
        )

    # ---- 状況別の実測打率・得点期待値（球団を問わず全体） ----
    st.subheader('📐 状況別 実測打率・得点期待値')
    st.caption('全対戦相手を合わせた実際のプレーバイプレーから集計（標本数が少ない場面は N/A）')
    st.dataframe(
        df_situational_stats.style.background_gradient(
            subset=['打率', '得点期待値'], cmap='RdYlGn',
        ),
        width='stretch', height=320,
    )

    st.divider()

    # ---- 全24場面 ----
    vs_label = opponent if pitcher == ALL_P else f"{opponent}・{tp.loc[pitcher, '選手名']}"
    st.subheader(f'📋 {batter} vs {vs_label} — 全 24 場面')
    df_all = predict_type_all(re24, type_model, batter_id, weights, batter_form, pitcher_form)
    tab_chart, tab_table = st.tabs(['グラフ', 'テーブル'])
    with tab_chart:
        st.plotly_chart(plot_24_bar(df_all, batter, vs_label), width='stretch')
    with tab_table:
        df_show = df_all[['アウト','ランナー','打者wOBA','投手補正','相性補正','対戦wOBA予測','基礎RE24','補正後期待得点']]
        st.dataframe(
            df_show.style.background_gradient(
                subset=['基礎RE24','補正後期待得点'], cmap='RdYlGn', vmin=0,
            ),
            width='stretch', height=600,
        )
        st.download_button(
            '📥 CSV ダウンロード',
            data=df_show.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'{batter}_vs_{vs_label}_期待得点.csv',
            mime='text/csv',
        )

    st.divider()

    # ---- 選手個人成績 ----
    st.subheader(f'👤 {batter} の個人成績')

    render_batter_stats(batter, df_woba, df_career, df_team_batting_stats, df_season)

    # ---- 投手の成績 ----
    page_pitcher_section(df_pitchers, opponent)


# ============================================================
# 打者・投手データの更新処理（個別ボタン・一括ボタンの共通部分）
#   取得 → 検証 → GitHub へ保存 を行い、(成功したか, メッセージ) を返す。
#   一括更新で片方が失敗しても、もう片方を続けられるよう例外は外に出さない。
# ============================================================
BATTER_REQUIRED_COLS  = ['区分種別', '選手名', '区分名', '球団', '試合', '打席',
                         '安打', '2B', '3B', '本塁', '四球', '死球', '敬遠', '犠打']
PITCHER_REQUIRED_COLS = ['選手名', '選手名_key', 'TM', '区分種別', '区分名',
                         '投球回', '被打者', '被安', '三振', '四球', '防御率']
SHRINK_LIMIT = 0.8   # 既存データより選手数がこの割合を下回ったら、不完全な取得とみなす


def _check_batter_df(df_new: pd.DataFrame, df_old: pd.DataFrame | None) -> list[str]:
    """取得した打者データに問題がないか調べ、問題の説明を返す（空なら問題なし）"""
    problems = []
    missing = [c for c in BATTER_REQUIRED_COLS if c not in df_new.columns]
    if missing:
        return [f'必要な列がありません: {", ".join(missing)}（サイト構造の変更、または表の取りこぼしの可能性）']

    # 球団の欠け: 2026-11-19 に一部球団の取得失敗が黙って保存され、12球団→4球団になったことへの対策
    for msg in df_new.attrs.get('failed_teams', []):
        problems.append(f'取得できなかった球団があります: {msg}（このまま保存すると、その球団の打者が消えます）')
    expected = {t[0] for t in BATTER_TEAMS}
    got = set(df_new['球団'].dropna().unique())
    if got != expected:
        problems.append(f'球団がそろっていません（{len(got)}/{len(expected)}）。欠け: {", ".join(sorted(expected - got))}')
    n_fp = len(df_new.attrs.get('failed_players', []))
    if n_fp:
        problems.append(f'詳細ページを取得できなかった選手が {n_fp} 人います')

    # 四球・死球・敬遠・犠打は2つ目の表にある列。取りこぼすと wOBA が低く出る
    for c in ('四球', '死球', '敬遠', '犠打'):
        if pd.to_numeric(df_new[c], errors='coerce').notna().mean() < 0.9:
            problems.append(f'「{c}」列の欠損が多い（2つ目の表を取りこぼしている可能性）')

    if df_old is not None and '選手名' in df_old.columns:
        n_old, n_new = df_old['選手名'].nunique(), df_new['選手名'].nunique()
        if n_old > 0 and n_new < n_old * SHRINK_LIMIT:
            problems.append(f'選手数が既存データ（{n_old} 人）より大きく減っています（{n_new} 人）。取得が途中で欠けた可能性')
    return problems


def _check_pitcher_df(df_new: pd.DataFrame, df_old: pd.DataFrame | None) -> list[str]:
    problems = []
    failed_teams = df_new.attrs.get('failed_teams', [])
    if failed_teams:
        problems.append(f'取得できなかった球団があります: {", ".join(failed_teams)}（このまま保存すると、その球団の投手が消えます）')

    missing = [c for c in PITCHER_REQUIRED_COLS if c not in df_new.columns]
    if missing:
        problems.append(f'必要な列がありません: {", ".join(missing)}')
        return problems

    keys_new = df_new[['選手名_key', 'TM']].drop_duplicates().shape[0]
    n_failed = len(df_new.attrs.get('failed_players', []))
    if keys_new and n_failed / keys_new > 0.3:
        problems.append(f'個別ページを取得できなかった投手が多い（{n_failed}/{keys_new} 人）。対戦相手別・球場別が欠けます')

    if df_old is not None and {'選手名_key', 'TM'}.issubset(df_old.columns):
        keys_old = df_old[['選手名_key', 'TM']].drop_duplicates().shape[0]
        if keys_old > 0 and keys_new < keys_old * SHRINK_LIMIT:
            problems.append(f'投手数が既存データ（{keys_old} 人）より大きく減っています（{keys_new} 人）')
    return problems


def _update_batters(gh_cfg: dict, sleep_sec: float) -> tuple[bool, str]:
    progress_bar = st.progress(0, text='[打者] 準備中...')
    log_area = st.empty()

    def on_progress(ti, n_teams, team_name, pi, n_players, player_name):
        pct = (ti + pi / max(n_players, 1)) / n_teams
        progress_bar.progress(min(pct, 1.0), text=f'[打者][{team_name}] {player_name} ({pi}/{n_players})')
        log_area.text(f'{team_name}: {player_name} を取得中 ({ti + 1}/{n_teams} 球団)')

    try:
        with st.spinner('打者データを取得中...'):
            df_new = scrape_all_batters(sleep_sec=sleep_sec, on_progress=on_progress)
    except Exception as e:
        return False, f'打者データの取得中にエラーが出ました: {type(e).__name__}: {e}'
    progress_bar.progress(1.0, text='[打者] 取得完了')
    log_area.empty()

    if df_new is None or df_new.empty:
        return False, 'データを取得できませんでした。サイト構造が変わっている可能性があります。'

    n_players = df_new['選手名'].nunique() if '選手名' in df_new.columns else 0
    st.caption(f'取得結果: {n_players} 選手 / {len(df_new):,} 行')
    try:
        _, lg = build_batter_woba(df_new)
        st.caption(f'取得データで計算したリーグ平均 wOBA: {lg:.3f}（極端に低い・高い場合は取得に問題がある可能性）')
    except Exception as e:
        st.caption(f'リーグ平均 wOBA を計算できませんでした（列の不足など）: {type(e).__name__}')

    try:
        df_old = load_batter_csv_from_github(gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
    except Exception:
        df_old = None

    problems = _check_batter_df(df_new, df_old)
    if problems:
        for msg in problems:
            st.warning(msg)
        if not st.session_state.get('force_save', False):
            return False, '検証で問題が見つかったため、GitHub のデータは更新していません（上の警告を確認してください）。'
        st.info('「検証で問題が出ても保存する」がオンなので保存します。')

    try:
        ok, msg = save_batter_csv_to_github(
            gh_cfg['token'], gh_cfg['repo_name'], df_new, gh_cfg['branch']
        )
    except Exception as e:
        return False, f'GitHub への保存中にエラーが出ました: {type(e).__name__}: {e}'
    if not ok:
        return False, f'GitHub への保存に失敗しました: {msg}'
    return True, f'{n_players} 選手分・{len(df_new):,} 行を GitHub に保存しました（{msg}）'


def _update_pitchers(gh_cfg: dict, sleep_sec: float) -> tuple[bool, str]:
    p_bar = st.progress(0, text='[投手] 準備中...')
    p_log = st.empty()

    def on_progress(ti, n_teams, team_name, pi, n_players, player_name):
        pct = (ti + pi / max(n_players, 1)) / n_teams
        p_bar.progress(min(pct, 1.0), text=f'[投手][{team_name}] {player_name} ({pi + 1}/{n_players})')
        p_log.text(f'{team_name}: {player_name} を取得中 ({ti + 1}/{n_teams} 球団)')

    try:
        with st.spinner('投手データを取得中...'):
            df_new = scrape_all_pitchers(sleep_sec=sleep_sec, on_progress=on_progress)
    except Exception as e:
        return False, f'投手データの取得中にエラーが出ました: {type(e).__name__}: {e}'
    p_bar.progress(1.0, text='[投手] 取得完了')
    p_log.empty()

    failed_players = df_new.attrs.get('failed_players', [])
    fail_errors    = df_new.attrs.get('errors', [])
    if failed_players:
        with st.expander(f'個別ページを取得できなかった投手（{len(failed_players)} 人・通算成績のみ保存）'):
            st.write('、'.join(failed_players))
    if fail_errors:
        with st.expander('失敗の理由（原因調査用）', expanded=df_new.empty):
            for msg in fail_errors:
                st.text(msg)

    if df_new.empty:
        return False, 'データを取得できませんでした。「失敗の理由」を確認してください。'

    n_p = df_new[['選手名_key', 'TM']].drop_duplicates().shape[0]
    st.caption(f'取得結果: {n_p} 投手 / {len(df_new):,} 行')

    try:
        df_old = load_pitcher_csv_from_github(gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
    except Exception:
        df_old = None

    problems = _check_pitcher_df(df_new, df_old)
    if problems:
        for msg in problems:
            st.warning(msg)
        if not st.session_state.get('force_save', False):
            return False, '検証で問題が見つかったため、GitHub のデータは更新していません（上の警告を確認してください）。'
        st.info('「検証で問題が出ても保存する」がオンなので保存します。')

    try:
        ok, msg = save_pitcher_csv_to_github(
            gh_cfg['token'], gh_cfg['repo_name'], df_new, gh_cfg['branch']
        )
    except Exception as e:
        return False, f'GitHub への保存中にエラーが出ました: {type(e).__name__}: {e}'
    if not ok:
        return False, f'GitHub への保存に失敗しました: {msg}'
    return True, f'{n_p} 投手分・{len(df_new):,} 行を GitHub に保存しました（{msg}）'


def _commit_with_progress(gh_cfg: dict, collected: list[dict]) -> tuple[int, int]:
    """試合ファイルを GitHub にコミットし、進捗を表示する"""
    commit_bar = st.progress(0, text='コミット準備中...')

    def on_progress(i, total, fname, success, msg):
        commit_bar.progress(i / total, text=f'[{i}/{total}] {"✅" if success else "❌"} {fname}')

    return commit_game_files(
        token=gh_cfg['token'], repo_name=gh_cfg['repo_name'],
        files=collected, branch=gh_cfg['branch'], on_progress=on_progress,
    )


def section_legacy_refetch(gh_cfg: dict):
    """旧形式（投手ID・打者ID の列がない）試合ファイルを取り直す"""
    st.subheader('🔄 既存試合の再取得（打者ID・対戦投手の列を追加）')
    st.caption(
        '2026-11 より前に取得した `_details.csv` には、打者ID・対戦投手（投手名・投手ID）の列がありません。'
        '打者タイプ × 投手タイプの分析にはこの列が必要なため、一球速報を取り直して上書きします。'
        '1 回の実行で取り直す件数を指定できます（1 試合あたり 約 2 秒 ＋ GitHub への保存 2 ファイル）。'
        '全件を一度に取り直す場合は、ローカルで `python scraper.py --refetch` を実行して git でコミットする方が速く確実です。'
    )
    c1, c2 = st.columns(2)
    with c1:
        n_max = st.number_input('今回取り直す試合数', min_value=1, max_value=900, value=100, key='legacy_n')
    with c2:
        sleep_sec = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 2.0, 0.5, key='legacy_sleep')

    if not st.button('🔄 旧形式の試合を再取得', key='legacy_refetch'):
        return

    with st.spinner('GitHub の既存データを確認中...'):
        legacy_ids, n_all = get_legacy_game_ids(gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
    if not legacy_ids:
        st.success(f'全 {n_all} 試合が新形式です。再取得は不要です。')
        return
    targets = legacy_ids[:int(n_max)]
    st.info(f'旧形式 **{len(legacy_ids)}** / 全 {n_all} 試合 → 今回 **{len(targets)}** 試合を再取得します。')

    bar = st.progress(0, text='準備中...')
    collected, n_ok, n_ng, n_nop = [], 0, 0, 0
    for i, r in enumerate(scrape_games(0, 0, sleep_sec=sleep_sec, game_ids=targets)):
        bar.progress((i + 1) / len(targets), text=f'[{i + 1}/{len(targets)}] {r["game_id"]} {r["card"]} {r["message"]}')
        if r['status'] != 'ok':
            n_ng += 1
            continue
        n_ok += 1
        n_nop += int(r['df_details']['投手ID'].isna().sum())
        gid, card = r['game_id'], r['card']
        collected.append({'filename': f'{gid}_{card}_details.csv', 'df': r['df_details']})
        if r['df_score'] is not None:
            collected.append({'filename': f'{gid}_{card}_score.csv', 'df': r['df_score']})

    st.caption(f'取得: 成功 {n_ok} / 失敗 {n_ng}　投手を特定できなかった打席: {n_nop}')
    if not collected:
        st.warning('保存するデータがありません。')
        return
    ok_cnt, fail_cnt = _commit_with_progress(gh_cfg, collected)
    if fail_cnt == 0:
        st.success(f'✅ {ok_cnt} ファイルを GitHub に保存しました。残りの旧形式: {len(legacy_ids) - n_ok} 試合')
    else:
        st.warning(f'{ok_cnt} 件成功 / {fail_cnt} 件失敗')
    reload_model()


def section_player_master(gh_cfg: dict):
    """選手マスタ（投打・球速）を更新する"""
    st.subheader('🧾 選手マスタ（投打・球速）の更新')
    st.caption(
        '一球速報に出てくる打者・投手について、スポナビの選手ページから **投打（右投左打など）** と '
        '**球種別の球速（ストレート平均球速・全球種の加重平均球速）** を取得し、`data/players.csv` に保存します。'
        '先に「既存試合の再取得」で打者ID・投手ID の列を揃えてください。'
        '球速は選手ページの「今シーズン」の集計値です。'
    )
    c1, c2 = st.columns(2)
    with c1:
        refetch_all = st.checkbox('取得済みの選手も取り直す（球速を最新にする）', key='players_all')
    with c2:
        sleep_sec = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 1.5, 0.5, key='players_sleep')

    if not st.button('🧾 選手マスタを更新', key='players_update'):
        return

    with st.spinner('試合データと既存の選手マスタを読み込み中...'):
        dfs, _ = load_details_from_github(gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
        try:
            df_old = load_players_csv_from_github(gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
        except Exception:
            df_old = None
    ids = collect_player_ids(dfs)
    if not ids:
        st.warning('打者ID・投手ID のある試合データがありません。先に「既存試合の再取得」を実行してください。')
        return
    if df_old is not None and not refetch_all:
        have = set(df_old['選手ID'].astype(str).str.replace(r'\.0$', '', regex=True))
        ids = [i for i in ids if i not in have]
    if not ids:
        st.success('新しく取得する選手はいません。')
        return

    st.info(f'{len(ids)} 人を取得します（約 {len(ids) * sleep_sec / 60:.0f} 分）。')
    bar = st.progress(0, text='準備中...')

    def on_progress(i, n, pid, name):
        bar.progress(i / n, text=f'[{i}/{n}] {pid} {name}')

    df_new = scrape_players(ids, sleep_sec=sleep_sec, on_progress=on_progress)
    failed = df_new.attrs.get('failed_ids', [])
    if failed:
        with st.expander(f'取得できなかった選手（{len(failed)} 人）'):
            st.text('\n'.join(failed))
    if df_new.empty:
        st.error('選手データを取得できませんでした。')
        return
    if len(failed) > len(ids) * 0.3 and not st.session_state.get('force_save', False):
        st.error('取得に失敗した選手が多いため保存を見送りました（サイト構造の変更の可能性）。')
        return

    df = merge_players(df_old, df_new)
    n_toda = df['投'].notna().sum()
    n_velo = df['直球_平均球速'].notna().sum()
    st.caption(f'マスタ {len(df)} 人（投打あり {n_toda} 人 / ストレート球速あり {n_velo} 人）')
    ok, msg = save_players_csv_to_github(gh_cfg['token'], gh_cfg['repo_name'], df, gh_cfg['branch'])
    if ok:
        st.success(f'✅ 選手マスタを保存しました（{msg}）')
    else:
        st.error(f'❌ 保存に失敗しました: {msg}')


# ============================================================
# 設定ページ: データ取得・更新
# ============================================================
def page_settings(gh_cfg):
    st.title('⚙️ 設定')
    st.caption('試合データの取得・更新を行います。')
    st.divider()

    if not gh_cfg:
        st.warning(
            'GitHub の認証情報が設定されていません。\n\n'
            '**Settings > Secrets** に以下を追加してください：\n\n'
            '```toml\n[github]\ntoken     = "ghp_..."\n'
            'repo_name = "your-name/repo"\nbranch    = "main"\n```'
        )
        return

    st.subheader('📥 試合データ取得・更新')

    col1, col2 = st.columns(2)
    with col1:
        count = st.number_input('取得件数（ID 連番）', min_value=1, max_value=900, value=200)
    with col2:
        sleep_sec = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 2.0, 0.5)

    if st.button('▶ 取得開始', type='primary'):

        with st.spinner('GitHub の既存データを確認中...'):
            complete_ids, incomplete_ids, latest_id = get_existing_game_ids(
                gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch']
            )

        scrape_start = latest_id + 1 if latest_id is not None else START_ID

        st.info(
            f'既存: 完全 **{len(complete_ids)}** 件 / 不完全 **{len(incomplete_ids)}** 件  \n'
            f'新規取得開始ID: **{scrape_start}** 〜 **{scrape_start + count - 1}**'
            + (f'  /  不完全データ **{len(incomplete_ids)}** 件を再取得' if incomplete_ids else '')
        )

        progress_bar  = st.progress(0, text='準備中...')
        status_area   = st.empty()
        log_container = st.container()
        collected: list[dict] = []
        ok, skip, err = 0, 0, 0

        # 不完全データの再取得
        for gid in sorted(incomplete_ids):
            for result in scrape_games(gid, 1, sleep_sec=sleep_sec):
                card = result['card'] or '---'
                if result['status'] == 'ok':
                    collected.append({'filename': f"{gid}_{card}_details.csv",
                                      'df': result['df_details']})
                    if result['df_score'] is not None:
                        collected.append({'filename': f"{gid}_{card}_score.csv",
                                          'df': result['df_score']})
                    ok += 1
                    with log_container:
                        st.text(f'🔄 {gid}  {card}  不完全データを再取得')
                else:
                    err += 1
                    with log_container:
                        st.text(f'⚠️ {gid}  {card}  {result["message"]}')

        # 新規IDのスクレイプ
        for i, result in enumerate(scrape_games(scrape_start, count, sleep_sec=sleep_sec)):
            pct  = (i + 1) / count
            gid  = result['game_id']
            card = result['card'] or '---'
            if result['status'] == 'ok':
                collected.append({'filename': f"{gid}_{card}_details.csv",
                                  'df': result['df_details']})
                if result['df_score'] is not None:
                    collected.append({'filename': f"{gid}_{card}_score.csv",
                                      'df': result['df_score']})
                ok += 1
                icon = '✅'
            elif result['status'] == 'no_game':
                skip += 1
                icon = '⬜'
            else:
                err += 1
                icon = '⚠️'
            progress_bar.progress(pct, text=f'[{i+1}/{count}] {gid} {card}')
            with log_container:
                st.text(f'{icon} {gid}  {card}  {result["message"]}')

        status_area.success(
            f'スクレイプ完了 — 取得: {ok} 件 / スキップ: {skip} 件 / エラー: {err} 件'
        )

        if not collected:
            st.info('新規・更新対象のデータがありませんでした。')
            return

        st.subheader('GitHub へ保存中...')
        commit_bar = st.progress(0, text='コミット準備中...')
        commit_log = st.container()

        def on_progress(i, total, fname, success, msg):
            commit_bar.progress(i / total, text=f'[{i}/{total}] {fname}')
            with commit_log:
                st.text(f'{"✅" if success else "❌"} {msg}')

        ok_cnt, fail_cnt = commit_game_files(
            token=gh_cfg['token'], repo_name=gh_cfg['repo_name'],
            files=collected, branch=gh_cfg['branch'], on_progress=on_progress,
        )

        if fail_cnt == 0:
            st.success(f'✅ {ok_cnt} ファイルを GitHub にコミットしました。')
        else:
            st.warning(f'{ok_cnt} 件成功 / {fail_cnt} 件失敗')

        st.info('データキャッシュをクリアして再計算します...')
        reload_model()

    st.divider()
    section_legacy_refetch(gh_cfg)

    st.divider()
    section_player_master(gh_cfg)

    st.divider()
    st.subheader('📁 データソース')
    st.markdown(
        '- 打席イベント: GitHub `data/gamedata/*_details.csv`（RE24 はレギュラーシーズンのみ。オールスター・ポストシーズンは除く）\n'
        '- 状況別打者成績: `data/all_batters_situational.csv`\n'
        '- 投手成績: `data/all_pitchers.csv`\n'
        '- 選手マスタ（投打・球速）: `data/players.csv`\n'
        '- 過去3年成績: `data/2023~2025打撃データ/stats_YYYY.csv`'
    )

    st.divider()
    st.subheader('🔁 打者・投手の成績をまとめて更新')
    st.caption(
        '打者の対戦成績と投手成績を続けて取得し、GitHub に保存します。'
        '合わせて十数分かかることがあります。片方が失敗してももう片方は続けます。'
        '取得データに問題（列の欠損、選手数の大幅な減少など）があると、既存データを壊さないよう保存を見送ります。'
    )
    bulk_sleep = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 1.5, 0.5, key='bulk_sleep')
    st.checkbox('検証で問題が出ても保存する（通常はオフ。個別の更新ボタンにも効きます）', key='force_save')

    if st.button('🔁 打者・投手をまとめて更新', type='primary', key='bulk_update'):
        results = []
        st.markdown('#### ① 打者')
        results.append(('打者',) + _update_batters(gh_cfg, bulk_sleep))
        st.markdown('#### ② 投手')
        results.append(('投手',) + _update_pitchers(gh_cfg, bulk_sleep))

        st.markdown('#### 結果')
        for name, ok, msg in results:
            if ok:
                st.success(f'✅ {name}: {msg}')
            else:
                st.error(f'❌ {name}: {msg}')
        if any(ok for _, ok, _ in results):
            load_model.clear()
            load_season_batting.clear()
            load_pitchers.clear()
            st.info('キャッシュをクリアしました。メイン・選手成績ページを開くと、最新のデータで再計算されます。')

    st.divider()
    st.subheader('👤 対戦成績データ（対戦相手別・球場別打率）の更新')
    st.caption(
        'nf3.sakura.ne.jp から全12球団・全選手の対戦相手別・球場別成績を再取得し、'
        'GitHub の `data/all_batters_situational.csv` を更新します。選手数が多いため数分かかります。'
    )
    batter_sleep = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 1.5, 0.5, key='batter_sleep')

    if st.button('🔄 対戦成績データを更新', type='secondary'):
        ok, msg = _update_batters(gh_cfg, batter_sleep)
        if ok:
            st.success(f'✅ {msg}')
            load_model.clear()
            load_season_batting.clear()
            st.info('キャッシュをクリアしました。メイン・選手成績ページを開くと、最新のデータで再計算されます。')
        else:
            st.error(f'❌ {msg}')

    st.divider()
    st.subheader('⚾ 投手成績データの更新')
    st.caption(
        'nf3.sakura.ne.jp から全12球団・全投手の通算・対戦相手別・球場別成績を再取得し、'
        'GitHub の `data/all_pitchers.csv` を更新します。投手数が多いため数分かかります。'
    )
    pitcher_sleep = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 1.5, 0.5, key='pitcher_sleep')

    if st.button('🔄 投手成績データを更新', type='secondary'):
        ok, msg = _update_pitchers(gh_cfg, pitcher_sleep)
        if ok:
            st.success(f'✅ {msg}')
            load_pitchers.clear()
        else:
            st.error(f'❌ {msg}')


# ============================================================
# メイン
# ============================================================
# 試合シミュレーションページ（game_sim.py）
# ============================================================
SIM_TEAMS = ['巨人', 'ヤクルト', 'ＤｅＮＡ', '中日', '阪神', '広島',
             'ソフトバンク', '日本ハム', 'オリックス', '楽天', '西武', 'ロッテ']


def _sim_team_inputs(sim, type_model, side: str, label: str, team_default: int, auto_form: bool):
    """片方のチームのスタメン・先発・調子の入力欄。{'team', 'lineup', 'sp', 'batter_forms', 'sp_form'} を返す"""
    teams = [t for t in SIM_TEAMS if t in sim['team_starters']] or sorted(sim['team_starters'])
    names = sim['names']
    forms = type_model.get('form')
    st.markdown(f'#### {label}')
    team = st.selectbox('球団', teams, index=min(team_default, len(teams) - 1), key=f'sim_team_{side}')

    sps = list(sim['team_starters'].get(team, {}))
    def _sp_label(pid):
        n = sim['team_starters'][team][pid]
        return f"{names.get(pid, pid)}（先発 {n} 試合）"
    sp = st.selectbox('先発投手', sps, format_func=_sp_label, key=f'sim_sp_{side}_{team}')
    sp_auto = (form_of(forms, '投手', sp) or {}).get('調子', '普通')
    sp_form = st.radio('先発の調子', FORM_LEVELS, index=FORM_LEVELS.index(sp_auto if auto_form else '普通'),
                       horizontal=True, key=f'sim_spform_{side}_{team}_{sp}_{auto_form}')

    batters = game_sim.team_batters(sim, team)
    default = game_sim.last_lineup(sim, team) or batters[:9]
    st.caption('打順（既定は今季最後の試合のスタメン）')
    lineup, bforms = [], []
    for i in range(9):
        c1, c2 = st.columns([3, 2])
        opts = batters if default[i] in batters else [default[i]] + batters
        with c1:
            pid = st.selectbox(f'{i + 1}番', opts, index=opts.index(default[i]),
                               format_func=lambda x: names.get(x, x), key=f'sim_bat_{side}_{team}_{i}')
        auto = (form_of(forms, '打者', pid) or {}).get('調子', '普通')
        with c2:
            f = st.selectbox('調子', FORM_LEVELS, index=FORM_LEVELS.index(auto if auto_form else '普通'),
                             key=f'sim_bform_{side}_{team}_{i}_{pid}_{auto_form}', label_visibility='collapsed')
        lineup.append(pid)
        bforms.append(f)
    return {'team': team, 'lineup': lineup, 'sp': sp, 'batter_forms': bforms, 'sp_form': sp_form}


def page_simulation(type_model):
    st.title('🎮 試合シミュレーション')
    if type_model is None or 'sim' not in type_model:
        st.error('シミュレーション用のモデルを作れませんでした（打席データに打者ID・投手ID の列がありません）。')
        return
    sim = type_model['sim']
    st.caption('スタメンと先発投手を入れると、1打席ずつの結果を確率で引いて試合を何度も繰り返し、勝率と予想スコアを出します。'
               '打者・投手の今季の成績（三振・四死球・単打・長打・本塁打の割合）、打者タイプ × 投手タイプの相性、'
               '調子を使います。先発は今季の先発での対戦打者数で降板し、その後はチームの救援陣が投げます。')

    auto_form = st.toggle('調子を直近の成績から自動で設定する（オフ = 全員「普通」）', value=False, key='sim_auto_form')
    ca, ch = st.columns(2)
    with ca:
        away = _sim_team_inputs(sim, type_model, 'away', 'ビジター（先攻）', 4, auto_form)
    with ch:
        home = _sim_team_inputs(sim, type_model, 'home', 'ホーム（後攻）', 0, auto_form)

    n = st.select_slider('シミュレーション回数', [200, 500, 1000, 2000, 5000], value=1000, key='sim_n')
    if away['team'] == home['team']:
        st.warning('同じ球団同士になっています。')
    if st.button('▶ シミュレーション開始', type='primary', key='sim_run'):
        with st.spinner(f'{n} 試合をシミュレーション中...'):
            st.session_state['sim_result'] = (game_sim.simulate_many(sim, away, home, n=n, seed=None), away, home)

    if 'sim_result' not in st.session_state:
        return
    r, away, home = st.session_state['sim_result']
    a, h = away['team'], home['team']
    st.divider()
    st.subheader('📊 結果')
    m1, m2, m3 = st.columns(3)
    m1.metric(f'{a} の勝率', f"{r['勝率']['away']:.1%}")
    m2.metric(f'{h} の勝率', f"{r['勝率']['home']:.1%}")
    m3.metric('引き分け（12回）', f"{r['勝率']['引き分け']:.1%}")
    s1, s2 = st.columns(2)
    s1.metric(f'{a} の平均得点', f"{r['平均得点']['away']:.2f}")
    s2.metric(f'{h} の平均得点', f"{r['平均得点']['home']:.2f}")

    dist = pd.DataFrame({a: r['得点分布']['away'], h: r['得点分布']['home']}).fillna(0).sort_index()
    fig = go.Figure([go.Bar(name=a, x=dist.index, y=dist[a]), go.Bar(name=h, x=dist.index, y=dist[h])])
    fig.update_layout(barmode='group', title='得点の分布', xaxis_title='得点', yaxis_title='割合',
                      yaxis_tickformat='.0%', height=360, margin=dict(l=20, r=20, t=50, b=40))
    st.plotly_chart(fig, width='stretch')

    top = r['スコア'].reset_index()
    top.columns = [a, h, '確率']
    top['確率'] = top['確率'].map(lambda v: f'{v:.1%}')
    st.markdown('**よく出たスコア**')
    st.dataframe(top, width='stretch', hide_index=True)

    inn = pd.DataFrame({a: r['イニング平均']['away'], h: r['イニング平均']['home']},
                       index=[f'{i}回' for i in range(1, 10)]).T.round(2)
    st.markdown('**イニングごとの平均得点**')
    st.dataframe(inn, width='stretch')

    st.divider()
    st.subheader('📝 1試合の流れ（シミュレーションの1回分）')
    smp = r['サンプル']
    cols = [f'{i}' for i in range(1, max(len(smp['line']['away']), len(smp['line']['home'])) + 1)]
    ls = pd.DataFrame([smp['line']['away'] + [None] * (len(cols) - len(smp['line']['away'])),
                       smp['line']['home'] + [None] * (len(cols) - len(smp['line']['home']))],
                      index=[a, h], columns=cols)
    ls = ls.map(lambda v: 'X' if v is None or (isinstance(v, float) and np.isnan(v)) else int(v))
    ls['計'] = [smp['away'], smp['home']]
    st.dataframe(ls, width='stretch')
    log = pd.DataFrame(smp['log'])
    log['攻撃'] = np.where(log['表裏'] == '表', a, h)
    st.dataframe(log[['回', '表裏', '攻撃', '打順', '打者', '投手', 'アウト', '走者', '結果', '得点', 'スコア']],
                 width='stretch', height=420, hide_index=True)
    st.caption('※ 結果は「三振・凡退・四死球・単打・二塁打・三塁打・本塁打」の7種類。走者の進み方（併殺・進塁打・盗塁など）は'
               '今季の一球速報で同じ状況・同じ結果のときに実際に起きたことから引いています。代打・守備交代・救援投手の個別の起用は'
               '入れていません。')


# ============================================================
def main():
    gh_cfg = get_github_cfg()

    # 起動時の初期ページを「メイン」に固定
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = '🔮 メイン（予測）'

    # ---- サイドバー: ナビゲーション ----
    with st.sidebar:
        st.markdown('## ⚾ NPB 得点期待値')
        st.divider()

        page_options = ['🔮 メイン（予測）', '🎮 試合シミュレーション', '👥 選手成績', '⚙️ 設定（データ更新）']
        try:
            default_idx = page_options.index(st.session_state['current_page'])
        except ValueError:
            default_idx = 0

        page = st.radio(
            'メニュー',
            page_options,
            index=default_idx,
            label_visibility='collapsed',
            key='sidebar_navigation',
        )
        st.session_state['current_page'] = page

        st.divider()
        st.caption('一球速報から計算した RE24 に、打者・投手の実力と打者タイプ × 投手タイプの相性を足して、'
                   '場面ごとの期待得点を推定します。')

    current = st.session_state['current_page']

    if current in ('🔮 メイン（予測）', '🎮 試合シミュレーション', '👥 選手成績'):
        if gh_cfg:
            src = (gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch'])
        else:
            # トークン未設定 → PC 内の data/ フォルダから読む（閲覧のみ。データ更新は不可）
            src = (None, None, None)
            st.sidebar.info('ローカルデータで表示中（GitHub トークン未設定）')
        (re24, counts, df_woba, league_avg, n_games, n_pa, df_career, df_team_runs,
         df_team_batting_stats, df_situational_stats, type_model) = load_model(*src, code_version=CODE_VERSION)
        df_pitchers = load_pitchers(*src, code_version=CODE_VERSION)
        df_season   = load_season_batting(*src, code_version=CODE_VERSION)

        if current == '🎮 試合シミュレーション':
            page_simulation(type_model)
        elif current == '🔮 メイン（予測）':
            if n_games == 0:
                st.info('試合データがありません。サイドバーから「⚙️ 設定」を選択して取得してください。')
                st.stop()
            page_main(re24, counts, df_woba, league_avg, n_games, n_pa, df_career, df_team_runs,
                      df_team_batting_stats, df_situational_stats, df_pitchers, df_season, type_model)
        else:
            # 選手成績ページは、試合データ（PBP）が未取得でも打者・投手の成績を閲覧できる
            page_players(df_woba, df_career, df_team_batting_stats, df_pitchers, df_season)

    elif current == '⚙️ 設定（データ更新）':
        page_settings(gh_cfg)


if __name__ == '__main__':
    main()
