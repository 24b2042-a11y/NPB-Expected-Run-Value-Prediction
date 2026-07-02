"""
app.py — NPB 得点期待値予測アプリ

構成:
  1. メインページ … 得点期待値予測 + 選手個人成績
  2. 設定ページ   … データ取得・更新
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from core import (
    build_model_from_dfs, predict_one, predict_all,
    get_player_career, get_player_current_team,
    RUNNER_LABEL, RUNNER_MAP,
)
from scraper import scrape_games
from github_sync import (
    commit_game_files,
    load_details_from_github,
    get_existing_game_ids,
)

st.set_page_config(
    page_title='NPB 得点期待値予測',
    page_icon='⚾',
    layout='wide',
    initial_sidebar_state='expanded',
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BATTER_CSV = os.path.join(BASE_DIR, 'data', 'all_batters_situational.csv')
STATS_DIR  = os.path.join(BASE_DIR, 'data', '2023~2025打撃データ')
START_ID   = 2021038624

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
@st.cache_data(show_spinner='データを読み込んでいます...')
def load_model(token: str, repo_name: str, branch: str):
    dfs, _ = load_details_from_github(token, repo_name, branch)
    return build_model_from_dfs(dfs, BATTER_CSV, STATS_DIR)


def reload_model():
    load_model.clear()
    st.rerun()


def get_github_cfg() -> dict | None:
    try:
        cfg = st.secrets['github']
        return {
            'token':     cfg['token'],
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
# メインページ: 得点期待値予測 + 選手個人成績
# ============================================================
def page_main(re24, counts, df_woba, league_avg, n_games, n_pa, df_career):
    st.title('⚾ 得点期待値予測')
    st.caption(f'使用データ: {n_games} 試合 / {n_pa:,} 打席  |  リーグ平均 wOBA: {league_avg:.3f}')
    st.divider()

    # ---- 選手名テキスト入力（インクリメンタルサーチ） ----
    all_batters = sorted(df_woba['選手名'].unique().tolist())
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

    st.divider()

    # ---- 1場面の予測結果 ----
    res = predict_one(re24, df_woba, league_avg, batter, opponent, out, runner)
    if res['note']:
        st.warning(res['note'])

    st.subheader('📊 予測結果')
    m1, m2, m3 = st.columns(3)
    m1.metric('基礎 RE24',      str(res['基礎RE24']       or 'N/A'))
    m2.metric('打者 wOBA',      str(res['打者wOBA']))
    delta_str = ''
    if res['補正後期待得点'] and res['基礎RE24']:
        delta_str = str(round(res['補正後期待得点'] - res['基礎RE24'], 3))
    m3.metric('補正後期待得点', str(res['補正後期待得点'] or 'N/A'), delta=delta_str)
    st.caption(f"対戦打席数: {res['対戦打席数']}  |  リーグ平均 wOBA: {league_avg:.3f}")

    st.divider()

    # ---- RE24 ヒートマップ ----
    st.subheader('🗺 RE24 行列')
    hl_runner = runner if runner != '走者なし' else ''
    st.plotly_chart(plot_re24_heatmap(re24, out, hl_runner), use_container_width=True)
    with st.expander('各マスのサンプル数'):
        st.dataframe(
            pd.DataFrame(counts, index=['0アウト','1アウト','2アウト'], columns=RUNNER_LABEL),
            use_container_width=True,
        )

    st.divider()

    # ---- 全24場面 ----
    st.subheader(f'📋 {batter} vs {opponent} — 全 24 場面')
    df_all = predict_all(re24, df_woba, league_avg, batter, opponent)
    tab_chart, tab_table = st.tabs(['グラフ', 'テーブル'])
    with tab_chart:
        st.plotly_chart(plot_24_bar(df_all, batter, opponent), use_container_width=True)
    with tab_table:
        df_show = df_all[['アウト','ランナー','打者wOBA','基礎RE24','補正後期待得点','対戦打席数']]
        st.dataframe(
            df_show.style.background_gradient(
                subset=['基礎RE24','補正後期待得点'], cmap='RdYlGn', vmin=0,
            ),
            use_container_width=True, height=600,
        )
        st.download_button(
            '📥 CSV ダウンロード',
            data=df_show.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'{batter}_vs_{opponent}_期待得点.csv',
            mime='text/csv',
        )

    st.divider()

    # ---- 選手個人成績 ----
    st.subheader(f'👤 {batter} の個人成績')

    tab_recent, tab_career = st.tabs(['今シーズン（対戦相手別）', '過去3年成績'])

    with tab_recent:
        df_b = (df_woba[df_woba['選手名'] == batter][['区分名','打席','wOBA']]
                .query('打席 >= 10').sort_values('wOBA', ascending=False))
        df_b['wOBA'] = df_b['wOBA'].round(3)
        df_b.columns = ['対戦相手','打席数','wOBA']
        if df_b.empty:
            st.caption('対戦相手別データ（打席10以上）がありません。')
        else:
            st.dataframe(df_b, use_container_width=True)

    with tab_career:
        career = get_player_career(df_career, batter)
        if career.empty:
            st.caption('過去成績データが見つかりません。')
        else:
            df_show = career[['年度','所属球団','試合','打席','打率','出塁率','長打率','OPS',
                              '本塁打','打点','盗塁']].reset_index(drop=True)
            st.dataframe(df_show, use_container_width=True)
            if len(career) >= 2:
                st.plotly_chart(plot_career_trend(career), use_container_width=True)


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
    st.subheader('📁 データソース')
    st.markdown(
        f'- 打席イベント: GitHub `data/gamedata/*_details.csv`\n'
        f'- 状況別打者成績: `data/all_batters_situational.csv`\n'
        f'- 過去3年成績: `data/2023~2025打撃データ/stats_YYYY.csv`'
    )


# ============================================================
# メイン
# ============================================================
def main():
    gh_cfg = get_github_cfg()

    # ---- サイドバー: ナビゲーション ----
    with st.sidebar:
        st.markdown('## ⚾ NPB 得点期待値')
        st.divider()
        page = st.radio(
            'メニュー',
            ['🔮 メイン（予測）', '⚙️ 設定（データ更新）'],
            label_visibility='collapsed',
        )
        st.divider()
        st.caption('PBP データから計算した RE24 × 打者 wOBA で場面ごとの期待得点を推定します。')

    if page == '🔮 メイン（予測）':
        if not gh_cfg:
            st.error('GitHub の認証情報が設定されていません。「⚙️ 設定」ページを確認してください。')
            st.stop()
        re24, counts, df_woba, league_avg, n_games, n_pa, df_career = load_model(
            gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch']
        )
        if n_games == 0:
            st.info('試合データがありません。サイドバーから「⚙️ 設定」を選択して取得してください。')
            st.stop()
        page_main(re24, counts, df_woba, league_avg, n_games, n_pa, df_career)

    elif page == '⚙️ 設定（データ更新）':
        page_settings(gh_cfg)


if __name__ == '__main__':
    main()
