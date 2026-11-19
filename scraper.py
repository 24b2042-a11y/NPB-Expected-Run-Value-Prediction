"""
scraper.py — スポナビ PBP スクレイパー（Streamlit / Colab 共通）
Google Drive / Colab 依存なし。結果を DataFrame で返す。

_details.csv の列:
    イニング, 打席順, 打順, 選手名, アウト, ランナー, 本文   … 従来からの列
    打者ID, 投手名, 投手ID, 攻撃                            … 2026-11 追加
      - 打者ID / 投手ID: スポナビの選手ID（/npb/player/{ID}/top）。
        選手マスタ（player_scraper.py → data/players.csv）と突き合わせて
        投打・球速などを付ける。名前の表記ゆれを避けるため ID で結合する。
      - 投手名 / 投手ID: その打席を「投げ終えた」投手。
        打席の途中で投手交代があった場合は交代後の投手になる
        （公式記録の責任投手の規則とは異なる場合がある）。
      - 攻撃: 'away'（表）または 'home'（裏）
"""
import re
import sys
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Generator

# ============================================================
# 定数
# ============================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

RUNNER_MAP = {
    '走者なし': '', '一塁': '1塁', '二塁': '2塁', '三塁': '3塁',
    '一二塁': '1,2塁', '一三塁': '1,3塁', '二三塁': '2,3塁', '満塁': '満塁',
}
OUT_MAP = {'無': 0, '一': 1, '二': 2}

RE_INNING = re.compile(r'^(\d{1,2})回(表|裏)$')
RE_BATTER = re.compile(
    r'^(?:(\d+)番|代打)\s*(.+?)\s*([無一二]死)'
    r'(走者なし|満塁|[一二三]塁|[一二三][二三]塁)$'
)
RE_SEQ    = re.compile(r'^(\d+)：')
RE_UNSAFE = re.compile(r'[\\/:*?"<>| ]')
RE_PLAYER_ID = re.compile(r'/npb/player/(\d+)/')
RE_ATTACK    = re.compile(r'^(.+?)の攻撃$')

# 投手交代の書式（2026年シーズンの一球速報で確認したもの）
#   「投手交代: 床田 → 齊藤汰」
#   「ピッチャー 柳 に代わって 橋本 がマウンドにあがる」
# どちらも 1 つ目のリンクが降板投手、2 つ目のリンクが登板投手。
RE_PITCH_CHANGE = re.compile(r'^(投手交代[:：]|ピッチャー\s*\S+\s*に代わって)')

# 新しく追加した列（core.py 側で「あれば引き継ぐ」列として参照する）
EXTRA_COLS = ['打者ID', '投手名', '投手ID', '攻撃']


def _player_id(a_tag) -> str | None:
    """<a href="/npb/player/1600123/top"> から選手ID '1600123' を取り出す"""
    if a_tag is None:
        return None
    m = RE_PLAYER_ID.search(a_tag.get('href', ''))
    return m.group(1) if m else None


def _player_links(tag) -> list[tuple[str, str | None]]:
    """tag 内の選手リンクを [(表示名, 選手ID), ...] で返す"""
    return [(a.get_text(strip=True), _player_id(a))
            for a in tag.find_all('a', class_='bb-liveText__player')]


def _attack_teams(soup: BeautifulSoup) -> dict:
    """
    「1回表 中日の攻撃」「1回裏 広島の攻撃」の見出しから
    {'away': '中日', 'home': '広島'} を返す（球団名はスポナビの短縮表記）。
    """
    teams = {}
    for head in soup.find_all('header', class_='bb-liveText__head'):
        h1  = head.find('h1', class_='bb-liveText__inning')
        det = head.find('p', class_='bb-liveText__detail')
        if not (h1 and det):
            continue
        m_inn = RE_INNING.match(h1.get_text(strip=True))
        m_att = RE_ATTACK.match(det.get_text(strip=True))
        if m_inn and m_att:
            side = 'away' if m_inn.group(2) == '表' else 'home'
            teams.setdefault(side, m_att.group(1))
        if len(teams) == 2:
            break
    return teams


def _starting_pitchers(soup: BeautifulSoup, teams: dict) -> dict:
    """
    試合前情報の「先発ピッチャーは広島が…床田、中日が…柳」から
    {'home': (名前, ID), 'away': (名前, ID)} を返す。
    各リンクの直前のテキストに含まれる球団名で、どちらのチームの投手かを判定する。
    """
    starters = {}
    for p in soup.find_all('p', class_='bb-liveText__summary'):
        if not p.get_text(strip=True).startswith('先発ピッチャー'):
            continue
        preceding = ''
        for child in p.children:
            name = getattr(child, 'name', None)
            if name == 'a' and 'bb-liveText__player' in child.get('class', []):
                # 直前テキストの中で「最後に」出てくる球団名をその投手のチームとする
                best_side, best_pos = None, -1
                for side, team in teams.items():
                    pos = preceding.rfind(team)
                    if pos > best_pos:
                        best_side, best_pos = side, pos
                if best_side is not None:
                    starters[best_side] = (child.get_text(strip=True), _player_id(child))
                preceding = ''
            else:
                preceding += child.get_text() if hasattr(child, 'get_text') else str(child)
        break
    return starters


# ============================================================
# 1試合をパースして DataFrame を返す
# ============================================================
def _parse_recap(soup: BeautifulSoup) -> pd.DataFrame:
    events, current_inn, current_half = [], None, None

    teams = _attack_teams(soup)
    # 守備側（＝投げている側）ごとの現在の投手 (名前, ID)
    pitcher = {side: None for side in ('home', 'away')}
    pitcher.update(_starting_pitchers(soup, teams))

    for tag in soup.find_all(['h1', 'li']):
        if tag.name == 'h1' and 'bb-liveText__inning' in tag.get('class', []):
            m = RE_INNING.match(tag.get_text(strip=True))
            if m:
                current_inn, current_half = int(m.group(1)), m.group(2)
            continue

        if tag.name != 'li' or 'bb-liveText__item' not in tag.get('class', []):
            continue
        if current_inn is None:
            continue

        attack  = 'away' if current_half == '表' else 'home'
        defense = 'home' if attack == 'away' else 'away'

        # --- 投手交代を先に反映する ---
        # 投手交代は打者の行と同じ <li> の中、打席結果より前に書かれるため、
        # その打席の投手は「交代後」の投手になる。
        for p_chg in tag.find_all('p', class_='bb-liveText__summary--change'):
            if not RE_PITCH_CHANGE.match(p_chg.get_text(' ', strip=True)):
                continue
            links = _player_links(p_chg)
            if len(links) < 2:
                continue
            (old_name, old_id), new = links[0], links[1]
            # 降板投手の ID と一致する側を優先。一致しなければ現在の守備側とみなす
            side = next((s for s, cur in pitcher.items()
                         if cur is not None and old_id and cur[1] == old_id), defense)
            if pitcher[side] is None:
                # 先発が取れていなかった場合: それまでの打席に降板投手を後から埋める
                for ev in events:
                    if ev['攻撃'] != side and ev['投手ID'] is None:
                        ev['投手名'], ev['投手ID'] = old_name, old_id
            pitcher[side] = new

        p_bat    = tag.find('p',   class_='bb-liveText__batter')
        div_cont = tag.find('div', class_='bb-liveText__content')
        div_text = tag.find('div', class_='bb-liveText__text')
        if not (p_bat and div_cont and div_text):
            continue

        bat_str  = p_bat.get_text(strip=True)
        cont_str = div_cont.get_text(strip=True)
        full_str = div_text.get_text(strip=True)

        m_seq = RE_SEQ.match(cont_str)
        seq   = int(m_seq.group(1)) if m_seq else None

        m_bat = RE_BATTER.match(bat_str)
        if not m_bat:
            continue

        batting_order = int(m_bat.group(1)) if m_bat.group(1) else None
        name   = m_bat.group(2)
        out    = OUT_MAP.get(m_bat.group(3)[0], '?')
        runner = RUNNER_MAP.get(m_bat.group(4), m_bat.group(4))
        body   = full_str[len(bat_str):].strip()
        cur_p  = pitcher[defense] or (None, None)

        events.append({
            'イニング': f'{current_inn}回{current_half}',
            '打席順':   seq,
            '打順':     batting_order,
            '選手名':   name,
            'アウト':   out,
            'ランナー': runner,
            '本文':     body,
            '打者ID':   _player_id(p_bat.find('a', class_='bb-liveText__player')),
            '投手名':   cur_p[0],
            '投手ID':   cur_p[1],
            '攻撃':     attack,
        })

    return pd.DataFrame(events)


def _clean_card(title_text: str) -> str:
    """タイトルから対戦カード名を抽出してファイル名用に整形"""
    m = re.search(r'([^\s]+vs\.[^\s]+)', title_text)
    return RE_UNSAFE.sub('_', m.group(1)) if m else 'Match'


def has_player_columns(df: pd.DataFrame | None) -> bool:
    """_details.csv が打者ID・投手ID の列を持つ（2026-11 以降の形式）か"""
    return df is not None and {'打者ID', '投手ID'}.issubset(df.columns)


# ============================================================
# メイン: IDリストをスクレイプして結果をジェネレータで返す
# ============================================================
def scrape_games(
    start_id: int,
    count: int,
    sleep_sec: float = 2.0,
    game_ids: list[int] | None = None,
) -> Generator[dict, None, None]:
    """
    start_id から count 個の試合IDを巡回する。
    game_ids を渡した場合は start_id / count を無視し、そのIDだけを巡回する
    （既存試合の再取得用）。

    各試合について yield するdict:
        game_id   : int
        card      : str   対戦カード名（ファイル名用）
        df_details: pd.DataFrame | None
        df_score  : pd.DataFrame | None
        status    : 'ok' | 'no_game' | 'no_events' | 'error'
        message   : str
    """
    from io import StringIO

    ids = list(game_ids) if game_ids is not None else [start_id + i for i in range(count)]
    for gid in ids:
        url = f"https://baseball.yahoo.co.jp/npb/game/{gid}/text"
        result = dict(game_id=gid, card='', df_details=None,
                      df_score=None, status='', message='')

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.RequestException as e:
            result.update(status='error', message=f'通信エラー: {e}')
            yield result
            time.sleep(sleep_sec)
            continue

        if resp.status_code != 200:
            result.update(status='no_game',
                          message=f'HTTP {resp.status_code}')
            yield result
            time.sleep(sleep_sec)
            continue

        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, 'html.parser')

        title_text = soup.title.text.strip() if soup.title else ''
        if not title_text or 'エラー' in title_text or '見つかりません' in title_text:
            result.update(status='no_game', message='試合ページではない')
            yield result
            time.sleep(sleep_sec)
            continue

        card = _clean_card(title_text)
        result['card'] = card

        # スコアボード
        try:
            dfs = pd.read_html(StringIO(resp.text))
            if dfs:
                result['df_score'] = dfs[0]
        except Exception:
            pass

        # 打席イベント
        df_events = _parse_recap(soup)
        if df_events.empty:
            result.update(status='no_events',
                          message='打席イベントなし（テキスト速報未掲載）')
        else:
            n_nop = int(df_events['投手ID'].isna().sum())
            msg = f'{len(df_events)} 打席取得'
            if n_nop:
                msg += f'（投手不明 {n_nop} 打席）'
            result.update(status='ok', df_details=df_events, message=msg)

        yield result
        time.sleep(sleep_sec)


# ============================================================
# ローカル実行: 既存の data/gamedata を新形式で取り直す
#   python scraper.py --refetch            … 投手ID列のないファイルだけ再取得
#   python scraper.py --refetch --all      … 全ファイルを再取得
# 取得後は git で data/gamedata をコミットする（Streamlit から1ファイルずつ
# コミットすると 1,600 回以上の API 呼び出しになるため、一括再取得はローカル推奨）。
# ============================================================
def _refetch_local(data_dir: str, refetch_all: bool, sleep_sec: float) -> None:
    import glob
    import os

    targets = []
    for path in sorted(glob.glob(os.path.join(data_dir, '*_details.csv'))):
        gid = int(os.path.basename(path).split('_', 1)[0])
        if refetch_all:
            targets.append(gid)
            continue
        try:
            cols = pd.read_csv(path, encoding='utf-8-sig', nrows=0).columns
        except Exception:
            cols = []
        if not {'打者ID', '投手ID'}.issubset(cols):
            targets.append(gid)

    print(f'再取得対象: {len(targets)} 試合（間隔 {sleep_sec} 秒 → 約 {len(targets) * sleep_sec / 60:.0f} 分）')
    n_ok = n_ng = 0
    for i, r in enumerate(scrape_games(0, 0, sleep_sec=sleep_sec, game_ids=targets), 1):
        gid, card = r['game_id'], r['card']
        if r['status'] == 'ok':
            r['df_details'].to_csv(os.path.join(data_dir, f'{gid}_{card}_details.csv'),
                                   index=False, encoding='utf-8-sig')
            if r['df_score'] is not None:
                r['df_score'].to_csv(os.path.join(data_dir, f'{gid}_{card}_score.csv'),
                                     index=False, encoding='utf-8-sig')
            n_ok += 1
        else:
            n_ng += 1
        print(f'[{i}/{len(targets)}] {gid} {card} {r["status"]} {r["message"]}')
    print(f'完了: 成功 {n_ok} / 失敗 {n_ng}')


if __name__ == '__main__':
    import argparse
    import os

    ap = argparse.ArgumentParser(description='スポナビ一球速報の取得')
    ap.add_argument('--refetch', action='store_true', help='data/gamedata の既存試合を新形式で再取得')
    ap.add_argument('--all', action='store_true', help='投手ID列があるファイルも含めて全て再取得')
    ap.add_argument('--sleep', type=float, default=2.0, help='リクエスト間隔（秒）')
    args = ap.parse_args()

    if not args.refetch:
        ap.print_help()
        sys.exit(0)
    base = os.path.dirname(os.path.abspath(__file__))
    _refetch_local(os.path.join(base, 'data', 'gamedata'), args.all, args.sleep)
