"""
batter_scraper.py — 打者の対戦相手別・球場別成績スクレイピング（nf3.sakura.ne.jp）

2段階構成:
  1. get_player_list()    : 球団ごとの選手名 + 詳細URL 一覧を取得
  2. scrape_player_stats(): 選手の詳細ページから「対戦相手」「球場」区分別の
                             成績テーブル（対チーム別成績(リーグ)/(交流戦)/球場別成績）を取得

出力は既存の data/all_batters_situational.csv と同じ列構成の DataFrame。

取得に失敗した球団・選手は DataFrame の attrs に記録する（2026-11-19〜）。
    df.attrs['failed_teams']   : 選手一覧を取得できなかった、または選手が0人だった球団
    df.attrs['failed_players'] : 詳細ページを取得できなかった選手
以前は失敗を黙って飛ばしていたため、一部の球団が抜けたデータがそのまま保存された
（2026-11-19 の自動更新で12球団→4球団になった）。

ローカル実行（アプリを使わずに data/all_batters_situational.csv を更新する）:
    python batter_scraper.py                 … 取得 → チェック → 問題がなければ保存
    python batter_scraper.py --force         … チェックで問題が出ても保存する
    python batter_scraper.py --sleep 2.0     … リクエスト間隔（秒）
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

OUTPUT_COLUMNS = ['球団', '選手名', 'URL', '区分種別', '区分名'] + STAT_COLUMNS


def team_list_url(team_code: str, league: int) -> str:
    """球団の打撃成績一覧ページの URL"""
    return (f"https://nf3.sakura.ne.jp/php/stat_disp/stat_disp.php"
            f"?y=0&leg={league}&tm={team_code}&fp=0&dn=1&dk=0")


# ============================================================
# Step 1: 球団ごとの選手一覧（選手名 + 詳細URL）
# ============================================================
def parse_player_list(html: str) -> list[dict]:
    """打撃成績一覧ページの HTML から [{'選手名', '詳細URL'}, ...] を取り出す"""
    soup = BeautifulSoup(html, 'html.parser')
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
    return players


def get_player_list(team_code: str, league: int, sleep_sec: float = 1.0) -> list[dict]:
    """
    球団の選手名・詳細URL一覧を取得する。
    Returns: [{'選手名': str, '詳細URL': str}, ...]
    """
    resp = requests.get(team_list_url(team_code, league), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    players = parse_player_list(resp.text)

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


def parse_player_stats(html: str) -> list[dict]:
    """選手詳細ページの HTML から対戦相手別・球場別成績を取り出す"""
    return _parse_stat_tables(BeautifulSoup(html, 'html.parser'))


def scrape_player_stats(player_url: str, sleep_sec: float = 1.0) -> list[dict]:
    """
    選手の詳細ページから対戦相手別・球場別成績を取得する。
    Returns: [{区分種別, 区分名, 打率, 試合, ...}, ...]
    """
    resp = requests.get(player_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    records = parse_player_stats(resp.text)

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


def build_batter_df(all_rows: list[dict], failed_teams: list[str] | None = None,
                    failed_players: list[str] | None = None) -> pd.DataFrame:
    """取得した行のリストを data/all_batters_situational.csv と同じ列構成の DataFrame にする"""
    if not all_rows:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        df = pd.DataFrame(all_rows)
        for c in OUTPUT_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = _coerce_types(df[OUTPUT_COLUMNS])
    df.attrs['failed_teams'] = list(failed_teams or [])
    df.attrs['failed_players'] = list(failed_players or [])
    return df


# ============================================================
# 一括スクレイピング: 全12球団 × 全選手
# ============================================================
def scrape_all_batters(sleep_sec: float = 1.5, on_progress=None) -> pd.DataFrame:
    """
    全球団・全選手の対戦相手別・球場別成績を取得し、
    data/all_batters_situational.csv と同じ列構成の DataFrame を返す。
    取得に失敗した球団・選手は df.attrs['failed_teams'] / ['failed_players'] に入る。

    on_progress(team_idx, n_teams, team_name, player_idx, n_players, player_name)
    """
    all_rows, failed_teams, failed = [], [], []   # failed: [(球団名, 選手の dict, エラー名), ...]
    n_teams = len(TEAMS)

    for ti, (team_name, code, league) in enumerate(TEAMS):
        try:
            players = get_player_list(code, league, sleep_sec=sleep_sec)
        except Exception as e:
            players = []
            failed_teams.append(f'{team_name}（一覧の取得失敗: {type(e).__name__}）')
        else:
            if not players:
                failed_teams.append(f'{team_name}（選手が0人）')

        n_players = len(players)
        for pi, p in enumerate(players):
            try:
                stats = scrape_player_stats(p['詳細URL'], sleep_sec=sleep_sec)
            except Exception as e:
                stats = []
                failed.append((team_name, p, type(e).__name__))

            for s in stats:
                row = {'球団': team_name, '選手名': p['選手名'], 'URL': p['詳細URL']}
                row.update(s)
                all_rows.append(row)

            if on_progress:
                on_progress(ti, n_teams, team_name, pi + 1, n_players, p['選手名'])

    # 詳細ページの取得に失敗した選手は、最後に間隔を空けてもう一度だけ取り直す
    # （2026-11-26: 一時的な失敗で近本・源田・谷川原が抜けたまま保存されていた）
    failed_players = []
    for team_name, p, err in failed:
        try:
            stats = scrape_player_stats(p['詳細URL'], sleep_sec=sleep_sec * 2)
        except Exception as e:
            failed_players.append(f"{team_name} {p['選手名']}（{type(e).__name__}）")
            continue
        for s in stats:
            row = {'球団': team_name, '選手名': p['選手名'], 'URL': p['詳細URL']}
            row.update(s)
            all_rows.append(row)

    return build_batter_df(all_rows, failed_teams, failed_players)


# ============================================================
# 保存前のチェック（アプリの _check_batter_df と同じ考え方）
# ============================================================
def check_batter_df(df_new: pd.DataFrame, df_old: pd.DataFrame | None = None,
                    shrink_limit: float = 0.8) -> list[str]:
    """取得データの問題点を返す（空なら問題なし）"""
    problems = []
    for msg in df_new.attrs.get('failed_teams', []):
        problems.append(f'取得できなかった球団: {msg}')
    n_fp = len(df_new.attrs.get('failed_players', []))
    if n_fp:
        problems.append(f'詳細ページを取得できなかった選手: {n_fp} 人')

    expected = {t[0] for t in TEAMS}
    got = set(df_new['球団'].dropna().unique()) if '球団' in df_new else set()
    if got != expected:
        problems.append(f'球団がそろっていません（{len(got)}/12）。欠け: {", ".join(sorted(expected - got))}')

    for c in ('四球', '死球', '敬遠', '犠打'):
        if c in df_new and pd.to_numeric(df_new[c], errors='coerce').notna().mean() < 0.9:
            problems.append(f'「{c}」列の欠損が多い')

    if df_old is not None and '選手名' in df_old.columns and len(df_old):
        n_old, n_new = df_old['選手名'].nunique(), df_new['選手名'].nunique()
        if n_new < n_old * shrink_limit:
            problems.append(f'選手数が既存データ（{n_old} 人）より大きく減っています（{n_new} 人）')
    return problems


if __name__ == '__main__':
    import argparse
    import os

    ap = argparse.ArgumentParser(description='nf3 から打者の対戦相手別・球場別成績を取得する')
    ap.add_argument('--sleep', type=float, default=1.5, help='リクエスト間隔（秒）')
    ap.add_argument('--force', action='store_true', help='チェックで問題が出ても保存する')
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, 'data', 'all_batters_situational.csv')

    def _p(ti, nt, team, pi, npl, name):
        print(f'[{ti + 1}/{nt} {team}] {pi}/{npl} {name}')

    df = scrape_all_batters(sleep_sec=args.sleep, on_progress=_p)
    df_old = pd.read_csv(out_path, encoding='utf-8-sig') if os.path.exists(out_path) else None
    problems = check_batter_df(df, df_old)
    print(f'取得: {df["選手名"].nunique()} 選手 / {len(df):,} 行 / {df["球団"].nunique()} 球団')
    for msg in problems:
        print('⚠', msg)
    if problems and not args.force:
        raise SystemExit('問題があるため保存しませんでした（--force で保存）')
    # 改行は LF（Windows の既定の CRLF だと、中身が同じでも git の差分になる）
    df.to_csv(out_path, index=False, encoding='utf-8-sig', lineterminator='\n')
    print('保存:', out_path)
