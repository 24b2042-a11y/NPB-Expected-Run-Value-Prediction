"""
app.py — NPB 得点期待値予測 + 試合データ自動取得 Streamlit アプリ
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from core import (
    build_model_from_dfs, predict_one, predict_all,
    RUNNER_LABEL, RUNNER_MAP,
)
from scraper import scrape_games
from github_sync import commit_game_files, load_details_from_github

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title='NPB 得点期待値予測',
    page_icon='⚾',
    layout='wide',
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
BATTER_CSV = os.path.join(BASE_DIR, 'data', 'all_batters_situational.csv')

# ============================================================
# モデルのキャッシュ読み込み（GitHub API 経由）
# ============================================================
@st.cache_data(show_spinner='RE24 を計算しています...')
def load_model(token: str, repo_name: str, branch: str):
    dfs, n_files = load_details_from_github(token, repo_name, branch)
    return build_model_from_dfs(dfs, BATTER_CSV)


def reload_model():
    load_model.clear()
    st.rerun()


# ============================================================
# GitHub Secrets 取得
# ============================================================
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
# 可視化ヘルパー
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
    df = df_all.copy()
    df['場面'] = df['アウト'].astype(str) + 'out / ' + df['ランナー']
    df_p = df[df['補正後期待得点'].notna()]
    fig = px.bar(
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


# ============================================================
# タブ: 試合データ取得
# ============================================================
def tab_scraper(gh_cfg):
    st.header('📥 試合データ取得')

    if not gh_cfg:
        st.warning(
            'GitHub の認証情報が設定されていません。\n\n'
            'Streamlit Community Cloud の **Settings > Secrets** に以下を追加してください：\n\n'
            '```toml\n'
            '[github]\n'
            'token     = "ghp_..."\n'
            'repo_name = "your-name/baseball-re24"\n'
            'branch    = "main"\n'
            '```'
        )
        return

    START_ID = 2021038624

    st.subheader('取得範囲の設定')
    col1, col2 = st.columns(2)
    with col1:
        count = st.number_input('取得件数（ID 連番）', min_value=1, max_value=900, value=200)
    with col2:
        sleep_sec = st.slider('リクエスト間隔（秒）', 1.0, 5.0, 2.0, 0.5)

    st.caption(f'対象 ID: **{START_ID}** 〜 **{START_ID + count - 1}**（{count} 件）')

    if st.button('▶ 取得開始', type='primary'):

        progress_bar  = st.progress(0, text='準備中...')
        status_area   = st.empty()
        log_container = st.container()

        collected: list[dict] = []
        ok, skip, err = 0, 0, 0

        for i, result in enumerate(scrape_games(START_ID, count, sleep_sec=sleep_sec)):
            pct  = (i + 1) / count
            gid  = result['game_id']
            card = result['card'] or '---'

            if result['status'] == 'ok':
                fname = f"{gid}_{card}_details.csv"
                collected.append({'filename': fname, 'df': result['df_details']})
                if result['df_score'] is not None:
                    collected.append({
                        'filename': f"{gid}_{card}_score.csv",
                        'df': result['df_score'],
                    })
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
            st.info('取得できた試合データがありませんでした。')
            return

        st.subheader('GitHub へ保存中...')
        commit_bar = st.progress(0, text='コミット準備中...')
        commit_log = st.container()

        def on_progress(i, total, fname, success, msg):
            commit_bar.progress(i / total, text=f'[{i}/{total}] {fname}')
            with commit_log:
                st.text(f'{"✅" if success else "❌"} {msg}')

        ok_cnt, fail_cnt = commit_game_files(
            token       = gh_cfg['token'],
            repo_name   = gh_cfg['repo_name'],
            files       = collected,
            branch      = gh_cfg['branch'],
            on_progress = on_progress,
        )

        if fail_cnt == 0:
            st.success(f'✅ {ok_cnt} ファイルを GitHub にコミットしました。')
        else:
            st.warning(f'{ok_cnt} 件成功 / {fail_cnt} 件失敗')

        st.info('RE24 キャッシュをクリアして再計算します...')
        reload_model()


# ============================================================
# タブ: 得点期待値予測
# ============================================================
def tab_predict(re24, counts, df_woba, league_avg, n_games, n_pa):
    st.header('🔮 得点期待値予測')

    st.sidebar.header('🎯 場面設定')
    batters   = sorted(df_woba['選手名'].unique().tolist())
    batter    = st.sidebar.selectbox('打者', batters)
    opponents = sorted(df_woba['区分名'].unique().tolist())
    opponent  = st.sidebar.selectbox('対戦相手', opponents)
    out       = st.sidebar.radio('アウトカウント', [0, 1, 2],
                                  format_func=lambda x: f'{x} アウト',
                                  horizontal=True)

    st.sidebar.markdown('**ランナー状態**')
    c1, c2, c3 = st.sidebar.columns(3)
    r1 = c1.checkbox('1塁', key='r1')
    r2 = c2.checkbox('2塁', key='r2')
    r3 = c3.checkbox('3塁', key='r3')

    runner_map = {
        (False,False,False): '走者なし',
        (True, False,False): '1塁',
        (False,True, False): '2塁',
        (False,False,True):  '3塁',
        (True, True, False): '1,2塁',
        (True, False,True):  '1,3塁',
        (False,True, True):  '2,3塁',
        (True, True, True):  '満塁',
    }
    runner = runner_map[(r1, r2, r3)]
    st.sidebar.markdown(f'選択中: **{runner}**')

    col_info, col_pred = st.columns(2)

    with col_info:
        st.subheader('📊 データ概要')
        m1, m2, m3 = st.columns(3)
        m1.metric('試合数', f'{n_games}')
        m2.metric('総打席数', f'{n_pa:,}')
        m3.metric('平均 wOBA', f'{league_avg:.3f}')

        st.markdown(f'**{batter} の対戦相手別 wOBA**')
        df_b = (df_woba[df_woba['選手名'] == batter][['区分名','打席','wOBA']]
                .query('打席 >= 10').sort_values('wOBA', ascending=False))
        df_b['wOBA'] = df_b['wOBA'].round(3)
        df_b.columns = ['対戦相手','打席数','wOBA']
        st.dataframe(df_b, use_container_width=True, height=220)

    with col_pred:
        st.subheader('1 場面の予測')
        res = predict_one(re24, df_woba, league_avg, batter, opponent, out, runner)
        if res['note']:
            st.warning(res['note'])
        p1, p2, p3 = st.columns(3)
        p1.metric('基礎 RE24',      str(res['基礎RE24'] or 'N/A'))
        p2.metric('打者 wOBA',      str(res['打者wOBA']))
        delta_str = ''
        if res['補正後期待得点'] and res['基礎RE24']:
            delta_str = str(round(res['補正後期待得点'] - res['基礎RE24'], 3))
        p3.metric('補正後期待得点', str(res['補正後期待得点'] or 'N/A'), delta=delta_str)
        st.caption(f"対戦打席数: {res['対戦打席数']}  |  リーグ平均 wOBA: {league_avg:.3f}")

    st.divider()

    st.subheader('🗺 RE24 行列')
    hl_runner = runner if runner != '走者なし' else ''
    st.plotly_chart(plot_re24_heatmap(re24, out, hl_runner), use_container_width=True)

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

    with st.expander('RE24 各マスのサンプル数'):
        st.dataframe(
            pd.DataFrame(counts, index=['0アウト','1アウト','2アウト'],
                         columns=RUNNER_LABEL),
            use_container_width=True,
        )


# ============================================================
# メイン
# ============================================================
def main():
    st.title('⚾ NPB 得点期待値予測')
    st.caption('PBP データから計算した RE24 × 打者 wOBA で場面ごとの期待得点を推定します。')

    gh_cfg = get_github_cfg()

    tab1, tab2 = st.tabs(['📥 試合データ取得・更新', '🔮 得点期待値予測'])

    with tab1:
        tab_scraper(gh_cfg)

    with tab2:
        if not gh_cfg:
            st.error('GitHub の認証情報が設定されていません。')
            st.stop()
        re24, counts, df_woba, league_avg, n_games, n_pa = load_model(
            gh_cfg['token'], gh_cfg['repo_name'], gh_cfg['branch']
        )
        if n_games == 0:
            st.info('まだ試合データがありません。「📥 試合データ取得・更新」タブからデータを取得してください。')
            st.stop()
        tab_predict(re24, counts, df_woba, league_avg, n_games, n_pa)


if __name__ == '__main__':
    main()
