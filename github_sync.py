"""
github_sync.py — GitHub リポジトリへの CSV 自動コミット＆読み込み
Streamlit Cloud のファイルシステムは読み取り専用のため、
データの永続化と読み込みをすべて GitHub API 経由で行う。

【高速化】
data/gamedata/ 配下のファイルを1件ずつ decoded_content で取得すると
ファイル数分の API リクエストが発生し非常に遅くなる（N+1問題）。
そこでリポジトリの zipball を1回のリクエストで取得し、ローカルで
展開してから必要なファイルだけを読む方式に変更した。
"""
import io
import zipfile
import requests
import pandas as pd
from github import Github, GithubException


def get_repo(token: str, repo_name: str):
    return Github(token).get_repo(repo_name)


# ============================================================
# CSV を任意のパスに upsert（新規 or 上書き）
# ============================================================
def upsert_csv_at_path(
    repo,
    df: pd.DataFrame,
    path: str,
    commit_message: str,
    branch: str = 'main',
) -> tuple[bool, str]:
    content_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')

    try:
        try:
            existing = repo.get_contents(path, ref=branch)
            sha = existing.sha
        except GithubException:
            sha = None

        if sha:
            repo.update_file(path, commit_message, content_bytes, sha, branch=branch)
            action = '更新'
        else:
            repo.create_file(path, commit_message, content_bytes, branch=branch)
            action = '新規作成'

        return True, f'{action}: {path}'

    except GithubException as e:
        return False, f'GitHub エラー: {e.status} {e.data}'
    except Exception as e:
        return False, f'予期せぬエラー: {e}'


# ============================================================
# CSV を data/gamedata/ に upsert（新規 or 上書き）
# ============================================================
def upsert_csv(
    repo,
    df: pd.DataFrame,
    filename: str,
    commit_message: str,
    branch: str = 'main',
) -> tuple[bool, str]:
    path = f'data/gamedata/{filename}'
    return upsert_csv_at_path(repo, df, path, commit_message, branch)


# ============================================================
# 対戦相手別・球場別打撃成績CSV（all_batters_situational.csv）の
# GitHub への保存 / GitHub からの読み込み（単一ファイルなので
# get_contents + decoded_content で1回のAPIコールのみ）
# ============================================================
def save_batter_csv_to_github(
    token: str,
    repo_name: str,
    df: pd.DataFrame,
    branch: str = 'main',
    path: str = 'data/all_batters_situational.csv',
) -> tuple[bool, str]:
    repo = get_repo(token, repo_name)
    return upsert_csv_at_path(repo, df, path, '[auto] update all_batters_situational.csv', branch)


def load_batter_csv_from_github(
    token: str,
    repo_name: str,
    branch: str = 'main',
    path: str = 'data/all_batters_situational.csv',
) -> pd.DataFrame | None:
    """
    GitHub Contents API は1MB超のファイルだと content(base64) を返さない
    （encoding が 'none' になり decoded_content が使えない）ため、
    download_url 経由で raw content を直接取得する。
    """
    repo = get_repo(token, repo_name)
    try:
        content = repo.get_contents(path, ref=branch)
    except GithubException:
        return None

    if isinstance(content, list):
        # 想定外だが、ディレクトリを指してしまった場合は None を返す
        return None

    try:
        if content.encoding == 'base64':
            raw = content.decoded_content
        else:
            resp = requests.get(
                content.download_url,
                headers={'Authorization': f'token {token}'},
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.content
    except Exception:
        return None

    return pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')


# ============================================================
# 複数ファイルを一括コミット
# ============================================================
def commit_game_files(
    token: str,
    repo_name: str,
    files: list[dict],
    branch: str = 'main',
    on_progress=None,
) -> tuple[int, int]:
    repo = get_repo(token, repo_name)
    ok_count, fail_count = 0, 0

    for i, item in enumerate(files):
        fname   = item['filename']
        df      = item['df']
        success, message = upsert_csv(repo, df, fname, f'[auto] add {fname}', branch)
        if success:
            ok_count += 1
        else:
            fail_count += 1
        if on_progress:
            on_progress(i + 1, len(files), fname, success, message)

    return ok_count, fail_count


# ============================================================
# 【新規】リポジトリの zipball を1回のリクエストで取得し、
# data/gamedata/*_details.csv をメモリ上に展開して返す。
#
# 従来: ファイル数 N 件につき N 回の API コール（decoded_content の遅延取得）
# 改善: API コール 1 回（archive_link 取得）+ HTTP ダウンロード 1 回
# ============================================================
def _fetch_details_archive(
    token: str,
    repo_name: str,
    branch: str = 'main',
) -> list[tuple[str, pd.DataFrame]]:
    """
    Returns: [(filename, df), ...]  ※ *_details.csv のみ
    """
    repo = get_repo(token, repo_name)

    # get_archive_link はリダイレクト先URLを1回のAPIコールで取得するだけ
    archive_url = repo.get_archive_link('zipball', ref=branch)

    resp = requests.get(
        archive_url,
        headers={'Authorization': f'token {token}'},
        timeout=60,
    )
    resp.raise_for_status()

    results: list[tuple[str, pd.DataFrame]] = []

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # zip内のパスは "{owner}-{repo}-{sha}/data/gamedata/xxx.csv" の形式
        target_names = [
            n for n in zf.namelist()
            if '/data/gamedata/' in n and n.endswith('_details.csv')
        ]
        for name in target_names:
            try:
                with zf.open(name) as f:
                    df = pd.read_csv(f, encoding='utf-8-sig')
                fname = name.rsplit('/', 1)[-1]
                results.append((fname, df))
            except Exception:
                continue

    return results


# ============================================================
# GitHub から _details.csv を全件読み込んで DataFrame を返す
# ============================================================
def load_details_from_github(
    token: str,
    repo_name: str,
    branch: str = 'main',
) -> tuple[list[pd.DataFrame], int]:
    """
    data/gamedata/ 以下の *_details.csv を全件取得して DataFrame のリストを返す。
    Returns: (dfs, n_files)
    """
    dfs     = []
    n_files = 0

    for fname, df in _fetch_details_archive(token, repo_name, branch):
        df['game_id'] = fname.split('_', 1)[0]

        # ファイル名 "{game_id}_{card}_details.csv" からホーム/アウェイの
        # 長い球団名を抽出する（card は "HomeAway" 形式で "vs." を含む）
        card = fname.split('_', 1)[1].replace('_details.csv', '')
        if 'vs.' in card:
            home_raw, away_raw = card.split('vs.', 1)
        else:
            home_raw, away_raw = None, None
        df['home_team_raw'] = home_raw
        df['away_team_raw'] = away_raw

        dfs.append(df)
        n_files += 1

    return dfs, n_files


# ============================================================
# 既存ファイルの状態を取得（スキップ判定用）
# ============================================================
def get_existing_game_ids(
    token: str,
    repo_name: str,
    branch: str = 'main',
) -> tuple[set[int], set[int], int | None]:
    """
    data/gamedata/ 内の _details.csv を調べて以下を返す。

    Returns
    -------
    complete_ids   : 試合終了まで取得済みの game_id セット
    incomplete_ids : ファイルはあるが不完全（試合終了なし or 打席数<10）の game_id セット
    latest_id      : 取得できたファイルの中で最大の game_id（新規取得の開始点算出に使用）
    """
    complete_ids   = set()
    incomplete_ids = set()
    latest_id      = None

    files = _fetch_details_archive(token, repo_name, branch)

    for fname, df in files:
        try:
            game_id = int(fname.split('_', 1)[0])
        except Exception:
            continue

        if _is_complete(df):
            complete_ids.add(game_id)
        else:
            incomplete_ids.add(game_id)

        if latest_id is None or game_id > latest_id:
            latest_id = game_id

    return complete_ids, incomplete_ids, latest_id


def _is_complete(df: pd.DataFrame) -> bool:
    """
    _details.csv が完全な試合データかを判定する。
    完全の条件:
      - 打席数が 10 以上
      - 本文に「試合終了」を含む行がある
    """
    if df is None or len(df) < 10:
        return False
    return df['本文'].fillna('').str.contains('試合終了').any()
