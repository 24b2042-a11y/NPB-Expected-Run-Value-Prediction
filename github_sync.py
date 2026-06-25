"""
github_sync.py — GitHub リポジトリへの CSV 自動コミット
PyGithub を使用。Streamlit Secrets から認証情報を取得。
"""
import base64
import io
from typing import Optional

import pandas as pd
from github import Github, GithubException


# ============================================================
# GitHub クライアント初期化
# ============================================================
def get_repo(token: str, repo_name: str):
    """
    repo_name: 'username/repo-name' 形式
    """
    g = Github(token)
    return g.get_repo(repo_name)


# ============================================================
# CSV を data/ に upsert（新規作成 or 上書き）
# ============================================================
def upsert_csv(
    repo,
    df: pd.DataFrame,
    filename: str,
    commit_message: str,
    branch: str = 'main',
) -> tuple[bool, str]:
    """
    data/<filename> に df を UTF-8 BOM 付き CSV としてコミットする。

    Returns
    -------
    (success: bool, message: str)
    """
    path = f'data/gamedata/{filename}'
    content = df.to_csv(index=False, encoding='utf-8-sig')
    content_bytes = content.encode('utf-8-sig')

    try:
        # 既存ファイルの SHA を取得（更新の場合に必要）
        try:
            existing = repo.get_contents(path, ref=branch)
            sha = existing.sha
        except GithubException:
            sha = None  # 新規ファイル

        if sha:
            repo.update_file(
                path=path,
                message=commit_message,
                content=content_bytes,
                sha=sha,
                branch=branch,
            )
            action = '更新'
        else:
            repo.create_file(
                path=path,
                message=commit_message,
                content=content_bytes,
                branch=branch,
            )
            action = '新規作成'

        return True, f'{action}: {path}'

    except GithubException as e:
        return False, f'GitHub エラー: {e.status} {e.data}'
    except Exception as e:
        return False, f'予期せぬエラー: {e}'


# ============================================================
# 複数ファイルを一括コミット（進捗コールバック付き）
# ============================================================
def commit_game_files(
    token: str,
    repo_name: str,
    files: list[dict],          # [{'filename': str, 'df': DataFrame}, ...]
    branch: str = 'main',
    on_progress=None,           # callable(i, total, filename, ok, msg)
) -> tuple[int, int]:
    """
    Returns (成功数, 失敗数)
    """
    repo = get_repo(token, repo_name)
    ok_count, fail_count = 0, 0

    for i, item in enumerate(files):
        fname = item['filename']
        df    = item['df']
        msg   = f'[auto] add {fname}'

        success, message = upsert_csv(repo, df, fname, msg, branch)
        if success:
            ok_count += 1
        else:
            fail_count += 1

        if on_progress:
            on_progress(i + 1, len(files), fname, success, message)

    return ok_count, fail_count
