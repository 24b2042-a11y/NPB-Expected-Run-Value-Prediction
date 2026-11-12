"""
pitcher_scraper.py — nf3.sakura.ne.jp から全投手の成績を取得する

打者のスクレイパー（batter_scraper.py）と同じ流れで取得する:
  1. 球団ごとの「投手成績一覧（背番号順）」ページから、その球団の全投手と
     各投手の個別ページへのリンクを取得する
       https://nf3.sakura.ne.jp/php/stat_disp/stat_disp.php?y=0&leg={0|1}&tm={球団コード}&fp=1&dn=1&dk=0
  2. 投手ごとの個別ページ（例: /Central/G/p/11_stat.htm）から、
     対戦相手別（チーム）・球場別の成績表を取得する

出力は打者のCSV（all_batters_situational.csv）に合わせた縦持ちの形式:
  区分種別 = '通算' / '対戦相手' / '球場'
  区分名   = 通算なら '全体'、対戦相手なら球団名（core.normalize_team_name で統一）、
             球場なら球場名

HTMLの表は標準ライブラリ（html.parser）だけで読む。pandas.read_html は
lxml の有無・pandas のバージョン・入れ子のレイアウト用テーブルに左右されて
全件失敗することがあるため使わない。

注意:
  ・二塁打・三塁打の被弾数は無いため、被wOBAは厳密には計算できない。
    代わりに K% / BB% / HR/9 / FIP を算出する。
  ・対戦相手別・球場別の表には「被打者」が無いため、
    被打者 ≒ 3 × 投球回 + 被安 + 四球 + 死球 で推定する（'被打者_推定' = True）。
  ・個別ページの表のレイアウトは実サイトで未確認。表が見つからない投手は
    df.attrs['failed_players'] に入れてスキップし、理由を df.attrs['errors'] に残す。
"""
import re
import time
from functools import reduce
from html.parser import HTMLParser
from urllib.parse import urljoin

import pandas as pd
import requests

from core import normalize_name, normalize_team_name

SITE_ROOT = 'https://nf3.sakura.ne.jp/'
BASE_URL = SITE_ROOT + 'php/stat_disp/stat_disp.php'
HEADERS  = {'User-Agent': 'Mozilla/5.0 (academic research; NPB sabermetrics)'}

# (leg, nf3の球団コード, 球団名, リーグ名)
TEAMS = [
    (0, 'T',  '阪神',         'セ・リーグ'),
    (0, 'DB', 'DeNA',         'セ・リーグ'),
    (0, 'G',  '読売',         'セ・リーグ'),
    (0, 'D',  '中日',         'セ・リーグ'),
    (0, 'C',  '広島',         'セ・リーグ'),
    (0, 'S',  'ヤクルト',     'セ・リーグ'),
    (1, 'H',  'ソフトバンク', 'パ・リーグ'),
    (1, 'F',  '日本ハム',     'パ・リーグ'),
    (1, 'B',  'オリックス',   'パ・リーグ'),
    (1, 'E',  '楽天',         'パ・リーグ'),
    (1, 'L',  '西武',         'パ・リーグ'),
    (1, 'M',  'ロッテ',       'パ・リーグ'),
]

ROSTER_TEXT_COLS = {'背番', '名前', '腕', '最終登板', 'href'}

# 個別ページ内の表の先頭列の見出し → 区分種別
SPLIT_KEYS = {'チーム': '対戦相手', '球場': '球場'}

MAX_PLAYER_ERRORS = 5   # 投手ごとのエラー内容は、同種の失敗が多くても先頭の数件だけ残す


# ============================================================
# 共通ユーティリティ
# ============================================================
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


def _get(url: str, params: dict | None = None) -> requests.Response:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # nf3 は UTF-8。ヘッダーに charset が無い場合に ISO-8859-1 と誤判定されないようにする
    if resp.encoding is None or resp.encoding.lower() == 'iso-8859-1':
        resp.encoding = 'utf-8'
    return resp


def _page_title(html: str) -> str:
    m = re.search(r'<title>(.*?)</title>', html, flags=re.S | re.I)
    return re.sub(r'\s+', ' ', m.group(1)).strip()[:60] if m else '(titleなし)'


# ============================================================
# 標準ライブラリだけのHTML表パーサー
# ============================================================
class _TableParser(HTMLParser):
    """
    HTML中の全ての <table> を、行 → セル(テキスト, リンクURL) のリストとして集める。
    入れ子の表（レイアウト用テーブルの中の本体テーブル）は、内側・外側それぞれが
    別の表として集まる。閉じタグが省略された td / tr も、次の開始タグで閉じる。
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[tuple[str, str | None]]]] = []
        self._stack: list[dict] = []

    def _close_cell(self, t: dict) -> None:
        c = t['cell']
        if c is not None and t['row'] is not None:
            text = re.sub(r'\s+', ' ', ''.join(c['text'])).strip()
            t['row'].append((text, c['href']))
        t['cell'] = None

    def _close_row(self, t: dict) -> None:
        self._close_cell(t)
        if t['row']:
            t['rows'].append(t['row'])
        t['row'] = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self._stack.append({'rows': [], 'row': None, 'cell': None})
            return
        if not self._stack:
            return
        t = self._stack[-1]
        if tag == 'tr':
            self._close_row(t)
            t['row'] = []
        elif tag in ('td', 'th'):
            self._close_cell(t)
            if t['row'] is None:
                t['row'] = []
            t['cell'] = {'text': [], 'href': None}
        elif tag == 'a' and t['cell'] is not None and t['cell']['href'] is None:
            href = dict(attrs).get('href')
            if href:
                t['cell']['href'] = href
        elif tag == 'br' and t['cell'] is not None:
            t['cell']['text'].append(' ')

    def handle_endtag(self, tag):
        if not self._stack:
            return
        t = self._stack[-1]
        if tag in ('td', 'th'):
            self._close_cell(t)
        elif tag == 'tr':
            self._close_row(t)
        elif tag == 'table':
            self._close_row(t)
            self.tables.append(self._stack.pop()['rows'])

    def handle_data(self, data):
        if self._stack and self._stack[-1]['cell'] is not None:
            self._stack[-1]['cell']['text'].append(data)


def _parse_tables(html: str) -> list[list[list[tuple[str, str | None]]]]:
    p = _TableParser()
    p.feed(html)
    p.close()
    # </table> の閉じ忘れも回収する
    while p._stack:
        t = p._stack.pop()
        p._close_row(t)
        p.tables.append(t['rows'])
    return p.tables


def _norm_header(cells: list[tuple[str, str | None]]) -> list[str]:
    return [re.sub(r'\s', '', c[0]) for c in cells]


def _row_texts(row, width: int) -> list[str]:
    """行のセルのテキストを、見出しの列数に合わせて切り詰め／空文字で埋める"""
    texts = [c[0] for c in row]
    return (texts + [''] * width)[:width]


# ============================================================
# 1. 球団ページ: 全投手の一覧（通算成績 + 個別ページのURL）
# ============================================================
def _link_base(html: str, page_url: str) -> str:
    """
    相対リンクを解決する基準URLを返す。
    nf3 のページは <base href="https://nf3.sakura.ne.jp/"> を持ち、リンクは
    「Central/T/p/13_stat.htm」のようにそのベース基準の相対パスで書かれている。
    ページ自身のURL（/php/stat_disp/ 配下）を基準につなぐと存在しない
    /php/stat_disp/Central/... になり 404 になるため、<base> があればそれを使い、
    無い場合もサイトのルートを基準にする。
    """
    m = re.search(r'<base\b[^>]*\bhref\s*=\s*["\']([^"\']+)["\']', html, flags=re.I)
    return urljoin(page_url, m.group(1)) if m else SITE_ROOT


def _fetch_team_roster(leg: int, code: str) -> pd.DataFrame:
    resp = _get(BASE_URL, {'y': 0, 'leg': leg, 'tm': code, 'fp': 1, 'dn': 1, 'dk': 0})
    tables = _parse_tables(resp.text)
    link_base = _link_base(resp.text, resp.url)

    for rows in tables:
        for i, row in enumerate(rows):
            header = _norm_header(row)
            if '名前' in header and '被打者' in header:
                name_idx = header.index('名前')
                records = []
                for r in rows[i + 1:]:
                    texts = _row_texts(r, len(header))
                    if texts and re.sub(r'\s', '', texts[0]) == header[0]:
                        continue  # 末尾に再掲される見出し行
                    rec = dict(zip(header, texts))
                    href = r[name_idx][1] if name_idx < len(r) else None
                    rec['href'] = urljoin(link_base, href) if href else None
                    records.append(rec)
                if not records:
                    raise ValueError('投手一覧の表はあるが、データ行が0件')
                return pd.DataFrame(records)

    raise ValueError(
        f'投手一覧の表が見つかりません（表の数={len(tables)}、'
        f'HTMLの長さ={len(resp.text)}、title={_page_title(resp.text)}）'
    )


def _clean_roster(df: pd.DataFrame) -> pd.DataFrame:
    # 名前が空の行（先頭の「合計」行など）を除く
    df = df[df['名前'].astype(str).str.strip() != ''].copy()
    df = df[df['背番'].astype(str) != '合計']

    for c in df.columns:
        if c not in ROSTER_TEXT_COLS:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 登板がなく成績が空の行は除く
    df = df[df['被打者'].notna()].copy()
    df['名前'] = df['名前'].astype(str).str.strip()
    return df.reset_index(drop=True)


def _roster_total_row(r: pd.Series) -> dict:
    """球団ページの1行（通算成績）を、縦持ち形式の1行にする"""
    row = {
        '区分種別': '通算', '区分名': '全体',
        '試合': r.get('試合'), '先発': r.get('先発'), '救援': r.get('救援'),
        '投球回': _ip_to_float(r.get('回数')),
        '被安': r.get('被安'), '被本': r.get('被本'), '三振': r.get('三振'),
        '四球': r.get('四球'), '死球': r.get('死球'),
        '失点': r.get('失点'), '自責': r.get('自責'), '防御率': r.get('防御率'),
        '被打者': r.get('被打者'), '被打者_推定': False,
    }
    if '故意四' in r.index:
        row['故意四'] = r.get('故意四')
    return row


# ============================================================
# 2. 投手個別ページ: 対戦相手別・球場別
# ============================================================
def _parse_split_tables(html: str) -> list[dict]:
    """
    個別ページから「チーム」「球場」で始まる表を探し、縦持ちの行リストにする。
    同じ見出しの表が複数に分かれている場合（打者ページの2ブロック構造と同様）は、
    先頭列をキーに横結合する。表が1つも見つからなければ ValueError。
    """
    tables = _parse_tables(html)

    groups: dict[str, list[pd.DataFrame]] = {}
    for rows in tables:
        for i, row in enumerate(rows):
            header = _norm_header(row)
            if (header and header[0] in SPLIT_KEYS
                    and '被安' in header and '回数' in header):
                body = []
                for r in rows[i + 1:]:
                    if len(r) < 5:
                        break  # 表の終わり（注記行など）
                    texts = _row_texts(r, len(header))
                    if re.sub(r'\s', '', texts[0]) == header[0]:
                        continue
                    body.append(texts)
                if body:
                    groups.setdefault(header[0], []).append(pd.DataFrame(body, columns=header))
                break

    if not groups:
        raise ValueError(
            f'対戦相手別・球場別の表が見つかりません（表の数={len(tables)}、'
            f'title={_page_title(html)}）'
        )

    out: list[dict] = []
    for first, tabs in groups.items():
        def _merge(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
            extra = [c for c in b.columns if c not in a.columns]
            return a.merge(b[[first] + extra], on=first, how='left')

        merged = reduce(_merge, tabs)
        for c in merged.columns:
            if c != first:
                merged[c] = pd.to_numeric(merged[c], errors='coerce')

        kind = SPLIT_KEYS[first]
        for _, r in merged.iterrows():
            name = str(r[first]).strip()
            if not name or name in ('nan', '合計', '計', '通算'):
                continue

            ip = _ip_to_float(r.get('回数'))
            h, bb, hbp = r.get('被安'), r.get('四球'), r.get('死球')
            if pd.isna(ip) or (ip == 0 and (pd.isna(h) or (h + bb + hbp) == 0)):
                continue  # 登板なし

            out.append({
                '区分種別': kind,
                '区分名': normalize_team_name(name) if kind == '対戦相手' else name,
                '試合': r.get('試合'), '先発': r.get('先発'), '救援': r.get('救援'),
                '投球回': ip, '被安': h, '被本': r.get('被本'), '三振': r.get('三振'),
                '四球': bb, '死球': hbp, '失点': r.get('失点'), '自責': r.get('自責'),
                '防御率': r.get('防御率'),
                '被打者': 3 * ip + h + bb + hbp, '被打者_推定': True,
            })
    return out


# ============================================================
# 指標
# ============================================================
def add_pitcher_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    K% / BB% / HR/9 / WHIP / FIP を追加する。

    FIP = (13*被本 + 3*(四球 + 死球) - 2*三振) / 投球回 + 定数
    定数 = リーグ防御率 - (13*HR + 3*(BB+HBP) - 2*K)/IP
           （全投手の「通算」行の合計から求め、FIPの平均が防御率と揃うようにする）
    FIP の四球は故意四球を除く（通算行のみ故意四球が分かる。
    対戦相手別・球場別の行は四球をそのまま使う）。
    """
    df = df.copy()
    ibb = df['故意四'].fillna(0) if '故意四' in df.columns else 0
    bb_unint = df['四球'] - ibb

    df['K%']   = df['三振'] / df['被打者']
    df['BB%']  = df['四球'] / df['被打者']
    df['HR/9'] = df['被本'] * 9 / df['投球回']
    df['WHIP'] = (df['被安'] + df['四球']) / df['投球回']

    raw = 13 * df['被本'] + 3 * (bb_unint + df['死球']) - 2 * df['三振']

    tot = df['区分種別'] == '通算'
    ip_total = df.loc[tot, '投球回'].sum()
    if ip_total and ip_total > 0:
        lg_era    = 9 * df.loc[tot, '自責'].sum() / ip_total
        lg_core   = raw[tot].sum() / ip_total
        fip_const = lg_era - lg_core
    else:
        fip_const = 3.1  # フォールバック（一般的な目安）

    df['FIP'] = raw / df['投球回'] + fip_const
    return df


# ============================================================
# メイン: 球団 → 投手 → 個別ページ
# ============================================================
def scrape_all_pitchers(
    sleep_sec: float = 1.5,
    min_ip: float = 0.0,
    on_progress=None,
) -> pd.DataFrame:
    """
    12球団の全投手について、通算・対戦相手別・球場別の成績を取得して結合する。

    on_progress(ti, n_teams, team_name, pi, n_players, player_name) は
    scrape_all_batters と同じ形式。

    取得に失敗した球団・投手は飛ばし、戻り値の DataFrame の attrs に記録する:
      df.attrs['failed_teams']   : 取得できなかった球団
      df.attrs['failed_players'] : 個別ページを取得できなかった投手
      df.attrs['errors']         : 失敗の理由（原因調査用。投手分は先頭の数件のみ）
    min_ip を指定すると、通算の投球回がそれ未満の投手を除く。
    """
    all_rows: list[dict] = []
    failed_teams: list[str] = []
    failed_players: list[str] = []
    errors: list[str] = []
    n_teams = len(TEAMS)

    for ti, (leg, code, team_raw, league_name) in enumerate(TEAMS):
        team = normalize_team_name(team_raw)

        if on_progress:
            on_progress(ti, n_teams, team, 0, 1, '選手一覧')
        try:
            roster = _clean_roster(_fetch_team_roster(leg, code))
        except Exception as e:
            failed_teams.append(team)
            errors.append(f'{team}（球団ページ）: {type(e).__name__}: {e}')
            time.sleep(sleep_sec)
            continue

        n_players = len(roster)
        for pi, (_, r) in enumerate(roster.iterrows()):
            pname = r['名前']
            if on_progress:
                on_progress(ti, n_teams, team, pi, n_players, pname)

            base = {
                '選手名': pname, '選手名_key': normalize_name(pname),
                '背番': r.get('背番'), '腕': r.get('腕'),
                'TM': code, '球団': team, 'リーグ': league_name,
            }
            all_rows.append({**base, **_roster_total_row(r)})

            href = r.get('href')
            if not href:
                failed_players.append(f'{team} {pname}')
                if len(errors) < len(failed_teams) + MAX_PLAYER_ERRORS:
                    errors.append(f'{team} {pname}: 個別ページへのリンクが無い')
                continue

            try:
                for sp in _parse_split_tables(_get(href).text):
                    all_rows.append({**base, **sp})
            except Exception as e:
                failed_players.append(f'{team} {pname}')
                if len(errors) < len(failed_teams) + MAX_PLAYER_ERRORS:
                    errors.append(f'{team} {pname}（{href}）: {type(e).__name__}: {e}')
            time.sleep(sleep_sec)

    if not all_rows:
        out = pd.DataFrame()
        out.attrs['failed_teams'] = failed_teams
        out.attrs['failed_players'] = failed_players
        out.attrs['errors'] = errors
        return out

    df = pd.DataFrame(all_rows)
    df = add_pitcher_metrics(df)

    if min_ip > 0:
        tot = df[df['区分種別'] == '通算']
        ok = tot[tot['投球回'] >= min_ip]
        keep = set(zip(ok['選手名_key'], ok['TM']))
        mask = [(k, t) in keep for k, t in zip(df['選手名_key'], df['TM'])]
        df = df[mask].reset_index(drop=True)

    df.attrs['failed_teams'] = failed_teams
    df.attrs['failed_players'] = failed_players
    df.attrs['errors'] = errors
    return df
