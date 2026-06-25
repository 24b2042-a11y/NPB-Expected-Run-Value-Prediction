"""
app.py — 得点期待値予測 Streamlit アプリ
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from core import (
    build_model, predict_one, predict_all,
    RUNNER_LABEL, RUNNER_MAP,
)

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title='NPB 得点期待値予測',
    page_icon='⚾',
    layout='wide',
)

# ============================================================
# データパス（リポジトリ内 data/ フォルダ）
# ============================================================
BASE_DIR    = os.path.dirname(__file__)
DETAILS_DIR = os.path.join(BASE_DIR, 'data')
BATTER_CSV  = os.path.join(BASE_DIR, 'data', 'all_batters_situational.csv')

# ============================================================
# モデルのキャッシュ読み込み
# ============================================================
@st.cache_data(show_spinner='データを読み込んでいます...')
def load_model():
    return build_model(DETAILS_DIR, BATTER_CSV)

# ============================================================
# RE24 ヒートマップ
# ============================================================
def plot_re24_heatmap(re24: pd.DataFrame, highlight_out=None, highlight_runner=None):
    z    = re24.values.astype(float)
    text = np.where(np.isnan(z), 'N/A', np.round(z, 2).astype(str))

    # ハイライト用マーカーの重ね描き
    shapes = []
    if highlight_out is not None and highlight_runner is not None:
        r_idx = RUNNER_MAP.get(highlight_runner, 0)
        shapes.append(dict(
            type='rect',
            x0=r_idx - 0.5, x1=r_idx + 0.5,
            y0=highlight_out - 0.5, y1=highlight_out + 0.5,
            line=dict(color='red', width=3),
        ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=RUNNER_LABEL,
        y=['0アウト', '1アウト', '2アウト'],
        text=text,
        texttemplate='%{text}',
        colorscale='RdYlGn',
        showscale=True,
        colorbar=dict(title='期待得点'),
        zmin=0, zmax=np.nanmax(z) if not np.all(np.isnan(z)) else 1,
    ))
    fig.update_layout(
        title='RE24 得点期待値行列（アウト × ランナー状態）',
        xaxis_title='ランナー状態',
        yaxis_title='アウトカウント',
        shapes=shapes,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


# ============================================================
# 24場面棒グラフ
# ============================================================
def plot_24_bar(df_all: pd.DataFrame, batter_name: str, opponent: str):
    df = df_all.copy()
    df['場面'] = df['アウト'].astype(str) + 'アウト / ' + df['ランナー']
    df_plot = df[df['補正後期待得点'].notna()].copy()

    fig = px.bar(
        df_plot,
        x='場面',
        y='補正後期待得点',
        color='アウト',
        color_continuous_scale='Blues',
        text=df_plot['補正後期待得点'].round(3),
        title=f'{batter_name} vs {opponent} — 24場面の補正後期待得点',
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_tickangle=-45,
        height=420,
        margin=dict(l=20, r=20, t=50, b=120),
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


# ============================================================
# メイン UI
# ============================================================
def main():
    st.title('⚾ NPB 得点期待値予測')
    st.caption('PBP（打席イベント）データから計算した RE24 行列と打者 wOBA を組み合わせて場面ごとの期待得点を推定します。')

    # --- データ読み込み ---
    try:
        re24, counts, df_woba, league_avg, n_games, n_pa = load_model()
    except FileNotFoundError as e:
        st.error(f'データが見つかりません。`data/` フォルダを確認してください。\n\n{e}')
        st.stop()

    # --- サイドバー: 入力 ---
    st.sidebar.header('🎯 場面設定')

    # 打者選択
    batters = sorted(df_woba['選手名'].unique().tolist())
    batter_name = st.sidebar.selectbox('打者', batters, index=0)

    # 対戦相手
    opponents = sorted(df_woba['区分名'].unique().tolist())
    opponent  = st.sidebar.selectbox('対戦相手', opponents, index=0)

    # アウトカウント
    out = st.sidebar.radio('アウトカウント', [0, 1, 2],
                           format_func=lambda x: f'{x} アウト', horizontal=True)

    # ランナー状態（チェックボックス）
    st.sidebar.markdown('**ランナー状態**')
    col1, col2, col3 = st.sidebar.columns(3)
    r1 = col1.checkbox('1塁', key='r1')
    r2 = col2.checkbox('2塁', key='r2')
    r3 = col3.checkbox('3塁', key='r3')

    runner_str_map = {
        (False, False, False): '走者なし',
        (True,  False, False): '1塁',
        (False, True,  False): '2塁',
        (False, False, True):  '3塁',
        (True,  True,  False): '1,2塁',
        (True,  False, True):  '1,3塁',
        (False, True,  True):  '2,3塁',
        (True,  True,  True):  '満塁',
    }
    runner = runner_str_map[(r1, r2, r3)]
    st.sidebar.markdown(f'選択中: **{runner}**')

    # --- メインエリア ---
    # 上段: データ概要 + 1場面予測
    col_info, col_pred = st.columns([1, 1])

    with col_info:
        st.subheader('📊 データ概要')
        c1, c2, c3 = st.columns(3)
        c1.metric('読み込み試合数', f'{n_games} 試合')
        c2.metric('総打席数',       f'{n_pa:,} 打席')
        c3.metric('リーグ平均 wOBA', f'{league_avg:.3f}')

        # 打者の対戦相手別 wOBA
        st.markdown(f'**{batter_name} の対戦相手別 wOBA**')
        df_batter = df_woba[df_woba['選手名'] == batter_name][['区分名', '打席', 'wOBA']].copy()
        df_batter = df_batter[df_batter['打席'] >= 10].sort_values('wOBA', ascending=False)
        df_batter['wOBA'] = df_batter['wOBA'].round(3)
        df_batter.columns = ['対戦相手', '打席数', 'wOBA']
        st.dataframe(df_batter, use_container_width=True, height=220)

    with col_pred:
        st.subheader('🔮 1場面予測')
        result = predict_one(re24, df_woba, league_avg,
                             batter_name, opponent, out, runner)
        if result['note']:
            st.warning(result['note'])

        m1, m2, m3 = st.columns(3)
        m1.metric('基礎 RE24',       f"{result['基礎RE24'] or 'N/A'}")
        m2.metric('打者 wOBA',        f"{result['打者wOBA']}")
        m3.metric('補正後期待得点',   f"{result['補正後期待得点'] or 'N/A'}",
                  delta=f"{round(result['補正後期待得点'] - result['基礎RE24'], 3) if result['補正後期待得点'] and result['基礎RE24'] else ''}")

        st.caption(f"対戦打席数: {result['対戦打席数']} 打席  |  "
                   f"リーグ平均 wOBA: {league_avg:.3f}")

    st.divider()

    # 中段: RE24 ヒートマップ
    st.subheader('🗺 RE24 行列')
    st.plotly_chart(
        plot_re24_heatmap(re24,
                          highlight_out=out,
                          highlight_runner=runner if runner != '走者なし' else ''),
        use_container_width=True,
    )

    # 下段: 24場面一覧
    st.subheader(f'📋 {batter_name} vs {opponent} — 全 24 場面')
    tab_chart, tab_table = st.tabs(['グラフ', 'テーブル'])

    df_all = predict_all(re24, df_woba, league_avg, batter_name, opponent)

    with tab_chart:
        st.plotly_chart(plot_24_bar(df_all, batter_name, opponent),
                        use_container_width=True)

    with tab_table:
        df_show = df_all[['アウト', 'ランナー', '打者wOBA',
                           '基礎RE24', '補正後期待得点', '対戦打席数']].copy()
        st.dataframe(
            df_show.style.background_gradient(
                subset=['基礎RE24', '補正後期待得点'],
                cmap='RdYlGn', vmin=0,
            ),
            use_container_width=True,
            height=600,
        )

        csv = df_show.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            '📥 CSV ダウンロード',
            data=csv.encode('utf-8-sig'),
            file_name=f'{batter_name}_vs_{opponent}_期待得点.csv',
            mime='text/csv',
        )

    # サンプル数テーブル（折りたたみ）
    with st.expander('RE24 各マスのサンプル数'):
        df_counts = pd.DataFrame(
            counts,
            index=pd.Index(['0アウト', '1アウト', '2アウト']),
            columns=RUNNER_LABEL,
        )
        st.dataframe(df_counts, use_container_width=True)


if __name__ == '__main__':
    main()
