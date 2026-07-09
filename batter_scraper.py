"""
batter_scraper.py — 打者の対戦相手別・球場別成績スクレイピング（nf3.sakura.ne.jp）

2段階構成:
  1. get_player_list()    : 球団ごとの選手名 + 詳細URL 一覧を取得
  2. scrape_player_stats(): 選手の詳細ページから「対戦相手」「球場」区分別の
                             成績テーブル（対チーム別成績(リーグ)/(交流戦)/球場別成績）を取得

出力は既存の data/all_batters_situational.csv と同じ列構成の DataFrame。
"""
import re
import time
import unicodedata
import requests
import pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# (表示用球団名, tm パラメータ, リーグ: 0=Central / 1=Pacific)
TEAMS = [
    ('阪神',         'T',  0),
    ('横浜DeNA',     'DB', 0),
    ('読売',         'G',  0),
    ('中日',         'D',  0),
    ('広島',         'C',  0),
    ('東京ヤクルト', 'S',  0),
    ('福岡ソフトバンク', 'H', 1),
    ('北海道日本ハム',   'F', 1),
    ('オリックス',       'B', 1),
    ('東北楽天',         'E', 1),
    ('埼玉西武',         'L', 1),
    ('千葉ロッテ',       'M', 1),
]

# 対チーム別成績・球場別成績テーブルの数値列（サイト表示順）
STAT_COLUMNS = [
    '打率', '試合', '打席', '打数', '得点', '安打', '2B', '3B', '本塁', '塁打', '打点',
    '三振', '四球', '敬遠', '死球', '犠打', '犠飛', '盗塁', '盗塁死', '失策',
    '出塁率', '長打率', 'OPS',
]

INT_COLUMNS   = ['試合', '打席', '打数', '得点', '安打', '2B', '3B', '本塁', '塁打', '打点',
                 '三振', '四球', '敬遠', '死球', '犠打', '犠飛', '盗塁', '盗塁死', '失策']
FLOAT_COLUMNS = ['打率', '出塁率', '長打率', 'OPS']


# ============================================================
# Step 1: 球団ごとの選手一覧（選手名 + 詳細URL）
# ============================================================
def get_player_list(team_code: str, league: int, sleep_sec: float = 1.0) -> list[dict]:
    """
    球団の選手名・詳細URL一覧を取得する。
    Returns: [{'選手名': str, '詳細URL': str}, ...]
    """
    url = (f"https://nf3.sakura.ne.jp/php/stat_disp/stat_disp.php"
           f"?y=0&leg={league}&tm={team_code}&fp=0&dn=1&dk=0")

    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, 'html.parser')

    players, seen = [], set()
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if len(cells) <= 15:
                continue
            name_cell = cells[1]
            link = name_cell.find('a')
            if not link:
                continue
            href = link.get('href')
            name = link.text.strip()
            if not href or name == '名前' or name in seen:
                continue
            clean_path = re.sub(r'^\.\./\.\./', '', href)
            full_url = f"https://nf3.sakura.ne.jp/{clean_path}"
            players.append({'選手名': name, '詳細URL': full_url})
            seen.add(name)

    if sleep_sec:
        time.sleep(sleep_sec)
    return players


# ============================================================
# Step 2: 選手詳細ページから「対戦相手」「球場」区分別成績を取得
# ============================================================
def _clean_cell(text: str) -> str:
    return unicodedata.normalize('NFKC', text.strip())


def _parse_stat_tables(soup: BeautifulSoup) -> list[dict]:
    """
    ページ内の全 <table> を走査し、1行目の見出しが
    'チーム' → 対チーム別成績(リーグ/交流戦どちらも同じ見出し構造)
    '球場'   → 球場別成績
    のテーブルだけを対象に、区分名ごとの成績行を抽出する。
    見出しでテーブルを識別するため、周辺の <h3> 等の文言が変わっても頑丈。
    """
    records = []

    for table in soup.find_all('table'):
        rows = table.find_all('tr')
        if not rows:
            continue

        header = [_clean_cell(c.get_text()) for c in rows[0].find_all(['td', 'th'])]
        if not header:
            continue

        if header[0] == 'チーム':
            kubun = '対戦相手'
        elif header[0] == '球場':
            kubun = '球場'
        else:
            continue

        for row in rows[1:]:
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if len(cells) < 1 + len(STAT_COLUMNS):
                continue
            label = _clean_cell(cells[0])
            record = {'区分種別': kubun, '区分名': label}
            for col, val in zip(STAT_COLUMNS, cells[1:1 + len(STAT_COLUMNS)]):
                record[col] = val
            records.append(record)

    return records


def scrape_player_stats(player_url: str, sleep_sec: float = 1.0) -> list[dict]:
    """
    選手の詳細ページから対戦相手別・球場別成績を取得する。
    Returns: [{区分種別, 区分名, 打率, 試合, ...}, ...]
    """
    resp = requests.get(player_url, headers=HEADERS, timeout=30)
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, 'html.parser')

    records = _parse_stat_tables(soup)

    if sleep_sec:
        time.sleep(sleep_sec)
    return records


# ============================================================
# 数値変換（"-" や "" は NaN/0 として扱う）
# ============================================================
def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace('-', pd.NA)
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    for col in FLOAT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ============================================================
# 一括スクレイピング: 全12球団 × 全選手
# ============================================================
def scrape_all_batters(sleep_sec: float = 1.5, on_progress=None) -> pd.DataFrame:
    """
    全球団・全選手の対戦相手別・球場別成績を取得し、
    data/all_batters_situational.csv と同じ列構成の DataFrame を返す。

    on_progress(team_idx, n_teams, team_name, player_idx, n_players, player_name)
    """
    all_rows = []
    n_teams = len(TEAMS)

    for ti, (team_name, code, league) in enumerate(TEAMS):
        try:
            players = get_player_list(code, league, sleep_sec=sleep_sec)
        except Exception:
            players = []

        n_players = len(players)
        for pi, p in enumerate(players):
            try:
                stats = scrape_player_stats(p['詳細URL'], sleep_sec=sleep_sec)
            except Exception:
                stats = []

            for s in stats:
                row = {'球団': team_name, '選手名': p['選手名'], 'URL': p['詳細URL']}
                row.update(s)
                all_rows.append(row)

            if on_progress:
                on_progress(ti, n_teams, team_name, pi + 1, n_players, p['選手名'])

    if not all_rows:
        return pd.DataFrame(columns=['球団', '選手名', 'URL', '区分種別', '区分名'] + STAT_COLUMNS)

    df = pd.DataFrame(all_rows)
    ordered_cols = ['球団', '選手名', 'URL', '区分種別', '区分名'] + STAT_COLUMNS
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = None
    df = df[ordered_cols]
    df = _coerce_types(df)
    return df
