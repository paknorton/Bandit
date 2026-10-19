#!/usr/bin/env python3

import subprocess

from pathlib import Path
from typing import Optional, Sequence, Union

from Bandit.exceptions import BanditError


class Repo:
    """Thin wrapper around a local git repository for querying revision info."""

    def __init__(self, path: Union[str, Path]):
        """Create a Repo for the git repository at ``path``.

        :param path: path to the local git repository
        :raises BanditError: if ``path`` does not exist or is not inside a git
                             work tree
        """

        self.path_repo = Path(path).resolve()

        if not self.path_repo.is_dir():
            raise BanditError(f'Not a directory: {self.path_repo}')

        result = self._git('rev-parse', '--is-inside-work-tree', allowed_codes=(0, 128))
        if result.returncode != 0 or result.stdout.strip() != 'true':
            raise BanditError(f'Not a git repository: {self.path_repo}')

    def _git(self,
             *args: str,
             allowed_codes: Sequence[int] = (0,)) -> subprocess.CompletedProcess:
        """Run a git command in the repository.

        :param args: arguments passed to the ``git`` executable
        :param allowed_codes: return codes that are not treated as errors
        :returns: the completed subprocess result
        :raises subprocess.CalledProcessError: if the return code is not allowed
        """

        result = subprocess.run(['git', *args],
                                cwd=self.path_repo,
                                capture_output=True,
                                text=True)

        if result.returncode not in allowed_codes:
            raise subprocess.CalledProcessError(result.returncode,
                                                result.args,
                                                output=result.stdout,
                                                stderr=result.stderr)
        return result

    @property
    def branch(self) -> Optional[str]:
        """Get the current branch.

        :returns: current branch name, or None when HEAD is detached
        """

        result = self._git('symbolic-ref', '--quiet', '--short', 'HEAD', allowed_codes=(0, 1))
        return result.stdout.rstrip() if result.returncode == 0 else None

    @property
    def commit(self) -> str:
        """Retrieve the current commit hash.

        :returns: the full commit hash for HEAD
        """

        return self._git('rev-parse', '--verify', 'HEAD').stdout.strip()

    @property
    def remote_url(self) -> Optional[str]:
        """Return the remote commit URL for the current branch or commit.

        This always returns a URL for the commit whether detached or associated
        with a branch. If it's a detached commit, remote origin is assumed.

        :returns: the remote commit URL, or None when the branch has no
                  configured upstream or its upstream is a local branch in this
                  same repository
        """

        branch = self.branch
        if branch is None:
            raw_url = self._git('config', '--get', 'remote.origin.url').stdout.rstrip()
        else:
            # `git config --get` exits 1 when the key is unset (e.g. a local-only
            # branch with no tracking remote); treat that as "no remote URL".
            remote_cfg = self._git('config', '--get', f'branch.{branch}.remote',
                                   allowed_codes=(0, 1))
            if remote_cfg.returncode == 1:
                return None  # Branch has no configured upstream remote.

            br_remote = remote_cfg.stdout.rstrip()
            if br_remote == '.':
                return None  # Upstream is another branch in this repository.
            raw_url = self._git('remote', 'get-url', br_remote).stdout.rstrip()

        base_url = self._normalize_remote(raw_url)
        return f'{base_url}/-/commit/{self.commit}'

    @staticmethod
    def _normalize_remote(raw_url: str) -> str:
        """Convert a git remote URL to an HTTPS web base URL.

        Handles SSH-style remotes (``git@host:group/repo.git``) and plain HTTPS
        remotes, stripping a trailing ``.git`` suffix.

        .. note::
            The commit-path suffix added by :attr:`remote_url` uses the GitLab
            ``/-/commit/<hash>`` convention. This assumes the parameter database
            is hosted on a GitLab-style server.

        :param raw_url: remote URL as reported by git
        :returns: an HTTPS base URL with any trailing ``.git`` removed
        """

        url = raw_url.removesuffix('.git')

        if url.startswith('git@'):
            # git@host:group/repo -> https://host/group/repo
            host, _, path = url[len('git@'):].partition(':')
            url = f'https://{host}/{path}'
        elif url.startswith('ssh://'):
            # ssh://git@host/group/repo -> https://host/group/repo
            remainder = url[len('ssh://'):]
            if '@' in remainder:
                remainder = remainder.split('@', 1)[1]
            url = f'https://{remainder}'

        return url

    def checkout(self, bct: str) -> None:
        """Checkout a branch, commit, or tag.

        Commits or tags always have HEAD detached.

        :param bct: branch name, commit hash, or tag to check out
        :raises BanditError: if ``bct`` is not a valid branch, commit, or tag
        """

        res = self._git('checkout', bct, allowed_codes=(0, 1))
        if res.returncode == 1:
            raise BanditError(f'Invalid branch, commit, or tag: {bct}')
