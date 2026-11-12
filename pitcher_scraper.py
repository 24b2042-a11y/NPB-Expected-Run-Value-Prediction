"""
pitcher_scraper.py — nf3.sakura.ne.jp から投手成績（セ・パ・交流戦）を取得する

取得元: 「投手成績一覧 総合成績」ページ
  https://nf3.sakura.ne.jp/php/stat_disp/stat_disp.php?y=0&leg={0|1|2}&tm=Sta&fp=1&dn=6&dk=0
  leg=0: セ・リーグ / leg=1: パ・リーグ / leg=2: 交流戦
  （y=0 は当年度。過去年度の指定方法は未確認）

打者のスクレイパー（1選手ずつ取得）と違い、リーグ単位の一覧表を
1回で取れるため、リクエストは最大3回で済む。

取得できる主な列:
  名前, TM(球団コード), 腕, 防御率, 試合, 先発, 救援, 回数, 被打者, 被安, 被本,
  三振, 四球, 故意四, 死球, 失点, 自責 など

注意: 二塁打・三塁打の被弾数は無いため、被wOBAは厳密には計算できない。
      代わりに K% / BB% / HR/9 / FIP を算出する。
"""
import io
import re
import time

import pandas as pd
import requests

from core import normalize_name

BASE_URL = 'https://nf3.sakura.ne.jp/php/stat_disp/stat_disp.php'
LEAGUES  = {0: 'セ・リーグ', 1: 'パ・リーグ', 2: '交流戦'}
HEADERS  = {'User-Agent': 'Mozilla/5.0 (academic research; NPB sabermetrics)'}

# nf3 の球団コード → 球団名（core.normalize_team_name で突き合わせ可能な表記）
TM_CODE_TO_TEAM = {
    'T': '阪神', 'DB': 'DeNA', 'G': '読売', 'D': '中日', 'C': '広島', 'S': 'ヤクルト',
    'H': 'ソフトバンク', 'F': '日本ハム', 'B': 'オリックス', 'E': '楽天',
    'L': '西武', 'M': 'ロッテ',
}

TEXT_COLS = {'名前', 'TM', '腕'}


def _ip_to_float(x) -> float:
    """
    投球回の表記（180.1 = 180と1/3回）を実数の回数に直す。
    小数点以下の .1 / .2 は 1/3, 2/3 回を意味するため、そのまま
    float として扱うと FIP や K/9 がずれる。
    """
    if pd.isna(x):
        return float('nan')
    x = round(float(x), 1)
    whole = int(x)
    tenth = int(round((x - whole) * 10))
    return whole + tenth / 3.0


def _fetch_league_table(leg: int, y: int = 0) -> pd.DataFrame:
    resp = requests.get(
        BASE_URL,
        params={'y': y, 'leg': leg, 'tm': 'Sta', 'fp': 1, 'dn': 6, 'dk': 0},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or 'utf-8'

    # 「被打者」を含むテーブル（= 投手成績一覧）だけを拾う
    tables = pd.read_html(io.StringIO(resp.text), match='被打者')
    df = tables[0]
    df.columns = [re.sub(r'\s', '', str(c)) for c in df.columns]
    return df


def _clean_table(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    # 表の末尾に見出し行が再掲されているので除外
    df = df[df['名前'] != '名前'].copy()
    df = df[df['名前'].notna()]

    for c in df.columns:
        if c not in TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.rename(columns={'名前': '選手名', '回数': '投球回_表記'})
    df['選手名']     = df['選手名'].astype(str).str.strip()
    df['選手名_key'] = df['選手名'].apply(normalize_name)
    df['球団']       = df['TM'].map(TM_CODE_TO_TEAM)
    df['リーグ']     = league_name
    df['投球回']     = df['投球回_表記'].apply(_ip_to_float)
    return df.reset_index(drop=True)


def add_pitcher_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    K% / BB% / HR/9 / FIP を追加する。

    FIP = (13*被本 + 3*(四球 + 死球) - 2*三振) / 投球回 + 定数
    定数 = リーグ防御率 - (13*HR + 3*(BB+HBP) - 2*K)/IP
           （取得した全投手の合計から求め、FIPの平均が防御率と揃うようにする）
    四球は故意四球を除いた数（nf3の「四球」は故意四球を含む前提）。
    """
    df = df.copy()
    ibb = df['故意四'].fillna(0) if '故意四' in df.columns else 0
    bb_unint = df['四球'] - ibb

    df['K%']   = df['三振'] / df['被打者']
    df['BB%']  = df['四球'] / df['被打者']
    df['HR/9'] = df['被本'] * 9 / df['投球回']

    raw = (13 * df['被本'] + 3 * (bb_unint + df['死球']) - 2 * df['三振'])

    ip_total = df['投球回'].sum()
    if ip_total and ip_total > 0:
        lg_era   = 9 * df['自責'].sum() / ip_total
        lg_core  = raw.sum() / ip_total
        fip_const = lg_era - lg_core
    else:
        fip_const = 3.1  # フォールバック（一般的な目安）

    df['FIP'] = raw / df['投球回'] + fip_const
    return df


def scrape_all_pitchers(
    sleep_sec: float = 1.5,
    y: int = 0,
    min_ip: float = 0.0,
    on_progress=None,
) -> pd.DataFrame:
    """
    セ・パ・交流戦の投手成績を取得して結合する。

    on_progress(i, total, league_name) を渡すと進捗を通知する。
    取得できなかったリーグはスキップし、全て失敗した場合は空の
    DataFrame を返す。
    """
    frames = []
    for i, (leg, league_name) in enumerate(LEAGUES.items()):
        if on_progress:
            on_progress(i, len(LEAGUES), league_name)
        try:
            raw = _fetch_league_table(leg, y=y)
            frames.append(_clean_table(raw, league_name))
        except Exception:
            pass
        time.sleep(sleep_sec)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = add_pitcher_metrics(df)
    if min_ip > 0:
        df = df[df['投球回'] >= min_ip].reset_index(drop=True)
    return df
