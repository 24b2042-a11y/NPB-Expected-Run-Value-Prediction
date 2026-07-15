"""
app.py — RE24 得点期待値予測シミュレーター（Streamlit UI - 完全版）
"""
import os
import glob
import pandas as pd
import streamlit as st

# core.py からロジックをインポート
from core import (
    build_model_from_dfs,
    predict_one,
    predict_all,
    RUNNER_LABEL
)

st.set_page_config(page_title="RE24得点期待値予測", layout="wide")

def load_local_data():
    """ローカルの data/ フォルダからCSVを自動ロードする"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return [], None
    
    # 状況別データ（CSV）の読み込み
    detail_files = glob.glob(os.path.join(data_dir, "details_*.csv"))
    if not detail_files:
        detail_files = glob.glob(os.path.join(data_dir, "*details*.csv"))
        
    dfs = []
    for f in detail_files:
        try:
            dfs.append(pd.read_csv(f, encoding='utf-8-sig'))
        except Exception as e:
            st.warning(f"ファイルの読み込みに失敗しました ({os.path.basename(f)}): {e}")

    # 選手状況別wOBAデータ（CSV）の読み込み
    batter_file = os.path.join(data_dir, "batter_woba.csv")
    batter_df = None
    if os.path.exists(batter_file):
        try:
            batter_df = pd.read_csv(batter_file, encoding='utf-8-sig')
        except Exception as e:
            st.warning(f"wOBAデータの読み込みに失敗しました (batter_woba.csv): {e}")
            
    return dfs, batter_df

def main():
    st.title("⚾ RE24 得点期待値予測システム")
    st.markdown("対戦相手（球団）と球場ごとのwOBA特性を考慮し、ベイズ収縮を用いて各シチュエーションの得点期待値を予測します。")

    # データのロード
    with st.spinner("データを読み込んでいます..."):
        dfs, batter_df = load_local_data()

    if not dfs:
        st.error("⚠️ `data/` フォルダ内に試合詳細CSV（`details_*.csv`）が見つかりません。")
        st.info("プロジェクトの `data` ディレクトリにCSVを配置してから再起動してください。")
        return

    # モデルのビルド
    stats_dir = "data"
    (re24, counts, df_woba, league_avg, n_games, n_pa, df_career,
     df_team_runs, df_team_batting_stats, df_situational_stats) = build_model_from_dfs(dfs, batter_df, stats_dir)

    # --------------------------------------------------------
    # メイン UI 制御
    # --------------------------------------------------------
    col1, col2 = st.columns([1, 3])

    with col1:
        st.header("🔍 条件設定")
        
        # 1. 打者の選択
        batters = sorted(df_woba['選手名'].unique().tolist())
        selected_batter = st.selectbox("打者を選択", batters)

        # 2. 対戦相手の選択
        opponents = sorted(df_woba[df_woba['区分'] == '対戦相手']['区分名'].unique().tolist())
        opponents = ["全体"] + opponents
        selected_opponent = st.selectbox("対戦相手を選択", opponents, index=0)

        # 3. 球場の選択
        stadiums = sorted(df_woba[df_woba['区分'] == '球場']['区分名'].unique().tolist())
        stadiums = ["全体"] + stadiums
        selected_stadium = st.selectbox("球場を選択", stadiums, index=0)

        st.write("---")
        st.subheader("特定シチュエーション予測")
        
        # 4. アウト数とランナー状況の選択
        selected_out = st.radio("アウト数", [0, 1, 2], format_func=lambda x: f"{x}アウト", horizontal=True)
        selected_runner = st.selectbox("ランナー状況", RUNNER_LABEL)

    with col2:
        # サマリー情報
        st.info(f"📊 分析データ規模: **{n_games}** 試合 / **{n_pa}** 打席 | リーグ平均wOBA: **{league_avg:.3f}**")

        # --------------------------------------------------------
        # A. 個別シチュエーション予測
        # --------------------------------------------------------
        st.header("🎯 シミュレーション結果")
        
        # predict_one の呼び出し（引数の不整合を完全に修正）
        res = predict_one(
            re24, df_woba, league_avg, 
            selected_batter, selected_opponent, selected_stadium, 
            selected_out, selected_runner
        )

        sub_col1, sub_col2, sub_col3 = st.columns(3)
        with sub_col1:
            st.metric(
                label=f"基礎RE24 ({selected_out}死 {selected_runner})", 
                value=f"{res['基礎RE24']:.3f} 点" if res['基礎RE24'] is not None else "データなし"
            )
        with sub_col2:
            st.metric(
                label="打者予測wOBA (ダブル補正後)", 
                value=f"{res['打者wOBA']:.3f}"
            )
        with sub_col3:
            st.metric(
                label="補正後 期待得点", 
                value=f"{res['補正後期待得点']:.3f} 点" if res['補正後期待得点'] is not None else "データなし",
                delta=f"{res['補正後期待得点'] - res['基礎RE24']:.3f} 点" if (res['補正後期待得点'] is not None and res['基礎RE24'] is not None) else None
            )

        if res['note']:
            st.caption(f"💡 *{res['note']}*")

        # --------------------------------------------------------
        # B. 24場面一覧予測（ヒートマップ形式のテーブル）
        # --------------------------------------------------------
        st.write("---")
        st.header("📋 24シチュエーション得点期待値マトリクス")
        st.write("打席完了までに、そのイニングで平均してあと何点入るかの予測値です。")

        # predict_all の呼び出し（引数の不整合を完全に修正）
        df_all_preds = predict_all(
            re24, df_woba, league_avg, 
            selected_batter, selected_opponent, selected_stadium
        )

        # 2Dピボットテーブルに整形して表示
        pivot_preds = df_all_preds.pivot(index='アウト', columns='ランナー', values='補正後期待得点')
        pivot_preds = pivot_preds.reindex(columns=RUNNER_LABEL)
        pivot_preds.index = [f"{i}アウト" for i in pivot_preds.index]

        st.dataframe(
            pivot_preds.style.background_gradient(cmap="Oranges", axis=None).format("{:.3f}"),
            use_container_width=True
        )

        # --------------------------------------------------------
        # C. 各種詳細データ
        # --------------------------------------------------------
        with st.expander("📊 選手情報 & 元データ統計"):
            tab1, tab2 = st.tabs(["球団別対戦成績", "全体シチュエーション成績"])
            
            with tab1:
                st.subheader(f"対戦球団別の実績スタッツ")
                df_p_team = df_team_batting_stats[df_team_batting_stats['選手名'] == selected_batter]
                if not df_p_team.empty:
                    st.dataframe(df_p_team, use_container_width=True, hide_index=True)
                else:
                    st.write("該当データがありません。")

            with tab2:
                st.subheader("状況（アウト・ランナー）別のリーグ実績統計")
                st.dataframe(df_situational_stats, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
