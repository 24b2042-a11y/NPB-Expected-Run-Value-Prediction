"""
player_scraper.py — スポナビ選手ページから選手マスタを作る

一球速報（_details.csv）の 打者ID / 投手ID をキーに、選手ページ
https://baseball.yahoo.co.jp/npb/player/{ID}/top から以下を取得する。

    選手ID, 選手名（フルネーム）, 選手名_key, 球団, 背番号, ポジション,
    投打, 投, 打,
    直球_平均球速, 直球_最高球速, 直球_割合, 球種数, 全球種_平均球速

    - 投打: 「右投左打」など。投 = 右/左、打 = 右/左/両
    - 直球_*: 球種「ストレート」の行（km/h, 割合は 0〜1）
    - 全球種_平均球速: 各球種の平均球速を投球割合で加重平均したもの
    - 球速データは選手ページに載っている「今シーズン」の集計値。
      予測の検証で学習/テストを時系列に分ける場合、テスト期間の投球も
      含んだ値である点に注意（タイプ分類程度なら影響は小さいと考えられる）。

保存先: data/players.csv（github_sync.save_players_csv_to_github）

ローカル実行:
    python player_scraper.py               … data/gamedata に出てくる選手のうち未取得分
    python player_scraper.py --from-teams  … 12球団の投手成績・打撃成績ページに載っている
                                             （今季1軍に出場した）選手のうち未取得分
    python player_scraper.py --all         … 全員を取り直す（シーズン途中の球速更新用）
"""
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

from scraper import HEADERS

PLAYER_URL = 'https://baseball.yahoo.co.jp/npb/player/{pid}/top'
RE_TODA    = re.compile(r'^([右左])投([右左両])打$')
RE_KANA    = re.compile(r'[（(].*?[）)]')

PLAYER_COLS = ['選手ID', '選手名', '選手名_key', '球団', '背番号', 'ポジション',
               '投打', '投', '打',
               '直球_平均球速', '直球_最高球速', '直球_割合', '球種数', '全球種_平均球速',
               '球種データ']
# 球種データ: 「球種:最高球速:平均球速:投球割合(%)」を ; で連結した生データ（後から別の指標を作れるように残す）

# スポナビの球団ID（/npb/teams/{ID}/...）。楽天だけ 376
TEAM_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 376]


def _to_float(s) -> float | None:
    try:
        return float(str(s).replace('%', '').replace(',', '').strip())
    except ValueError:
        return None


def _parse_pitch_table(soup: BeautifulSoup) -> dict:
    """「投球データ」表（球種・球速・投球割合）から球速の指標を作る"""
    out = {'直球_平均球速': None, '直球_最高球速': None, '直球_割合': None,
           '球種数': None, '全球種_平均球速': None, '球種データ': None}
    table = soup.find('table', class_='bb-pitchingRateTable')
    if table is None or table.tbody is None:
        return out

    rows = []
    for tr in table.tbody.find_all('tr'):
        th  = tr.find('th')
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if th is None or len(tds) < 3:
            continue
        # 列順: 最高球速, 平均球速, 投球割合(全体), 対左, 対右, ...
        rows.append({'球種': th.get_text(strip=True),
                     '最高': _to_float(tds[0]),
                     '平均': _to_float(tds[1]),
                     '割合': _to_float(tds[2])})
    if not rows:
        return out

    df = pd.DataFrame(rows)
    out['球種数'] = int(len(df))
    out['球種データ'] = ';'.join(
        f"{r['球種']}:{'' if r['最高'] is None else r['最高']}:{'' if r['平均'] is None else r['平均']}:{'' if r['割合'] is None else r['割合']}"
        for r in rows)
    st = df[df['球種'] == 'ストレート']
    if not st.empty:
        out['直球_平均球速'] = st['平均'].iloc[0]
        out['直球_最高球速'] = st['最高'].iloc[0]
        out['直球_割合']     = st['割合'].iloc[0] / 100 if st['割合'].iloc[0] is not None else None
    valid = df.dropna(subset=['平均', '割合'])
    if not valid.empty and valid['割合'].sum() > 0:
        out['全球種_平均球速'] = float((valid['平均'] * valid['割合']).sum() / valid['割合'].sum())
    return out


def parse_player_page(html: str, pid: str) -> dict:
    """選手ページの HTML から1行分の dict を返す"""
    from core import normalize_name

    soup = BeautifulSoup(html, 'html.parser')
    rec = {c: None for c in PLAYER_COLS}
    rec['選手ID'] = str(pid)

    h2 = soup.find('h2', class_='bb-profile__name')
    if h2 is not None:
        # 「床田 寛樹 （トコダ ヒロキ）」→「床田 寛樹」。ruby の rt（読み）は除く
        for rt in h2.find_all('rt'):
            rt.decompose()
        name = RE_KANA.sub('', h2.get_text(' ', strip=True)).strip()
        rec['選手名'] = re.sub(r'\s+', ' ', name)
        rec['選手名_key'] = normalize_name(rec['選手名'])

    num = soup.find('p', class_='bb-profile__number')
    pos = soup.find('p', class_='bb-profile__position')
    rec['背番号']     = num.get_text(strip=True) if num else None
    rec['ポジション'] = pos.get_text(strip=True) if pos else None

    for item in soup.find_all('div', class_='bb-profile__item'):
        dt, dd = item.find('dt'), item.find('dd')
        if dt and dd and dt.get_text(strip=True) == '投打':
            rec['投打'] = dd.get_text(strip=True)
            m = RE_TODA.match(rec['投打'])
            if m:
                rec['投'], rec['打'] = m.group(1), m.group(2)

    # タイトル「床田 寛樹 - 広島東洋カープ - プロ野球 - スポーツナビ」から球団名
    if soup.title:
        parts = [p.strip() for p in soup.title.get_text().split(' - ')]
        if len(parts) >= 2:
            rec['球団'] = parts[1]

    rec.update(_parse_pitch_table(soup))
    return rec


def scrape_players(player_ids, sleep_sec: float = 1.5, on_progress=None) -> pd.DataFrame:
    """
    選手IDのリストから選手マスタを作る。
    on_progress(i, n, pid, name) で進捗を通知する。
    取得できなかった ID は df.attrs['failed_ids'] に入れる。
    """
    ids = [str(p) for p in dict.fromkeys(player_ids) if pd.notna(p) and str(p).strip()]
    recs, failed = [], []
    for i, pid in enumerate(ids):
        n_before = len(recs)
        try:
            resp = requests.get(PLAYER_URL.format(pid=pid), headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                failed.append(f'{pid}: HTTP {resp.status_code}')
            else:
                resp.encoding = 'utf-8'
                rec = parse_player_page(resp.text, pid)
                if rec['選手名'] is None:
                    failed.append(f'{pid}: 選手名が見つからない（ページ構造の変更の可能性）')
                else:
                    recs.append(rec)
        except requests.RequestException as e:
            failed.append(f'{pid}: 通信エラー {e}')
        if on_progress:
            on_progress(i + 1, len(ids), pid, recs[-1]['選手名'] if len(recs) > n_before else '（失敗）')
        time.sleep(sleep_sec)

    df = pd.DataFrame(recs, columns=PLAYER_COLS)
    df.attrs['failed_ids'] = failed
    return df


def collect_player_ids(details_dfs: list[pd.DataFrame]) -> list[str]:
    """_details.csv 群に出てくる打者ID・投手IDを重複なしで返す"""
    ids = []
    for df in details_dfs:
        for col in ('打者ID', '投手ID'):
            if col in df.columns:
                ids.extend(df[col].dropna().astype('int64').astype(str).tolist()
                           if pd.api.types.is_numeric_dtype(df[col])
                           else df[col].dropna().astype(str).tolist())
    return list(dict.fromkeys(ids))


def collect_ids_from_teams(sleep_sec: float = 1.5) -> list[str]:
    """
    12球団の「投手成績」「打撃成績」ページ（/npb/teams/{ID}/pitchingstats, battingstats）
    に載っている選手の ID を返す。今季1軍に出場した選手が対象。
    試合データ（_details.csv）の再取得を待たずに選手マスタを作るときに使う。
    """
    ids = []
    for tid in TEAM_IDS:
        for kind in ('pitchingstats', 'battingstats'):
            url = f'https://baseball.yahoo.co.jp/npb/teams/{tid}/{kind}'
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                resp.encoding = 'utf-8'
                ids.extend(re.findall(r'/npb/player/(\d+)/', resp.text))
            time.sleep(sleep_sec)
    return list(dict.fromkeys(ids))


def merge_players(df_old: pd.DataFrame | None, df_new: pd.DataFrame) -> pd.DataFrame:
    """既存マスタに新規取得分を上書きマージする（選手IDで一意）"""
    if df_old is None or df_old.empty:
        return df_new.reset_index(drop=True)
    old = df_old.copy()
    old['選手ID'] = old['選手ID'].astype(str)
    old = old[~old['選手ID'].isin(df_new['選手ID'].astype(str))]
    return pd.concat([old, df_new], ignore_index=True)[PLAYER_COLS]


if __name__ == '__main__':
    import argparse
    import glob
    import os

    ap = argparse.ArgumentParser(description='スポナビ選手ページから data/players.csv を作る')
    ap.add_argument('--all', action='store_true', help='取得済みの選手も取り直す')
    ap.add_argument('--from-teams', action='store_true',
                    help='試合データではなく12球団の成績ページから選手IDを集める')
    ap.add_argument('--sleep', type=float, default=1.5, help='リクエスト間隔（秒）')
    args = ap.parse_args()

    base     = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, 'data', 'players.csv')
    if args.from_teams:
        ids = collect_ids_from_teams(args.sleep)
    else:
        dfs = [pd.read_csv(p, encoding='utf-8-sig')
               for p in glob.glob(os.path.join(base, 'data', 'gamedata', '*_details.csv'))]
        ids = collect_player_ids(dfs)
    if not ids:
        raise SystemExit('打者ID・投手ID のある _details.csv がありません。'
                         '先に python scraper.py --refetch を実行するか、--from-teams を付けてください。')

    # 選手ID・背番号は文字列で読む（数値で読むと育成選手の背番号「011」が「11」になる）
    df_old = (pd.read_csv(out_path, encoding='utf-8-sig', dtype={'選手ID': str, '背番号': str})
              if os.path.exists(out_path) else None)
    if df_old is not None and not args.all:
        ids = [i for i in ids if i not in set(df_old['選手ID'].astype(str))]
    print(f'取得対象: {len(ids)} 人（約 {len(ids) * args.sleep / 60:.0f} 分）')

    def _p(i, n, pid, name):
        print(f'[{i}/{n}] {pid} {name}')

    df_new = scrape_players(ids, sleep_sec=args.sleep, on_progress=_p)
    df = merge_players(df_old, df_new)
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'保存: {out_path}（{len(df)} 人）')
    for msg in df_new.attrs.get('failed_ids', []):
        print('失敗:', msg)
