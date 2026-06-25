"""
scraper.py — スポナビ PBP スクレイパー（Streamlit / Colab 共通）
Google Drive / Colab 依存なし。結果を DataFrame で返す。
"""
import re
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


# ============================================================
# 1試合をパースして DataFrame を返す
# ============================================================
def _parse_recap(soup: BeautifulSoup) -> pd.DataFrame:
    events, current_inn, current_half = [], None, None

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

        events.append({
            'イニング': f'{current_inn}回{current_half}',
            '打席順':   seq,
            '打順':     batting_order,
            '選手名':   name,
            'アウト':   out,
            'ランナー': runner,
            '本文':     body,
        })

    return pd.DataFrame(events)


def _clean_card(title_text: str) -> str:
    """タイトルから対戦カード名を抽出してファイル名用に整形"""
    m = re.search(r'([^\s]+vs\.[^\s]+)', title_text)
    return RE_UNSAFE.sub('_', m.group(1)) if m else 'Match'


# ============================================================
# メイン: IDリストをスクレイプして結果をジェネレータで返す
# ============================================================
def scrape_games(
    start_id: int,
    count: int,
    sleep_sec: float = 2.0,
) -> Generator[dict, None, None]:
    """
    start_id から count 個の試合IDを巡回する。

    各試合について yield するdict:
        game_id   : int
        card      : str   対戦カード名（ファイル名用）
        df_details: pd.DataFrame | None
        df_score  : pd.DataFrame | None
        status    : 'ok' | 'no_game' | 'no_events' | 'error'
        message   : str
    """
    from io import StringIO

    for i in range(count):
        gid = start_id + i
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
            result.update(status='ok', df_details=df_events,
                          message=f'{len(df_events)} 打席取得')

        yield result
        time.sleep(sleep_sec)
