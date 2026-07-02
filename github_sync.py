"""
github_sync.py — GitHub リポジトリへの CSV 自動コミット＆読み込み
Streamlit Cloud のファイルシステムは読み取り専用のため、
データの永続化と読み込みをすべて GitHub API 経由で行う。
"""
import io
import pandas as pd
from github import Github, GithubException


def get_repo(token: str, repo_name: str):
    return Github(token).get_repo(repo_name)


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
    path          = f'data/gamedata/{filename}'
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
    repo     = get_repo(token, repo_name)
    dfs      = []
    n_files  = 0

    try:
        contents = repo.get_contents('data/gamedata', ref=branch)
    except GithubException:
        return [], 0

    for item in contents:
        if not item.name.endswith('_details.csv'):
            continue
        try:
            raw      = item.decoded_content          # bytes
            df       = pd.read_csv(io.BytesIO(raw), encoding='utf-8-sig')
            game_id  = item.name.split('_', 1)[0]
            df['game_id'] = game_id
            dfs.append(df)
            n_files += 1
        except Exception:
            continue

    return dfs, n_files
