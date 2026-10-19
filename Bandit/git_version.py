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

    def resolve_commit(self, revision: str) -> Optional[str]:
        """Resolve a branch, commit, or tag to its full commit hash.

        A partial commit hash is accepted as long as it is unambiguous within
        the repository.

        :param revision: branch name, commit hash (full or partial), or tag
        :returns: the full commit hash, or None when ``revision`` does not
                  resolve to a commit in this repository
        """

        # `<rev>^{commit}` forces resolution to a commit object, so an
        # annotated tag is dereferenced to the commit it points at. git
        # rev-parse resolves an unambiguous partial hash to the full hash.
        result = self._git('rev-parse', '--verify', '--quiet', f'{revision}^{{commit}}',
                           allowed_codes=(0, 1))
        if result.returncode != 0:
            return None

        resolved = result.stdout.strip()
        return resolved if resolved else None

    def matches_revision(self, revision: Optional[str]) -> bool:
        """Check whether the repository is currently at the given revision.

        The ``revision`` may be a branch name, a tag, or a commit hash (full or
        partial). A partial commit hash matches when it is a prefix of the
        current commit hash (and, when it resolves through git, when it points
        at the current commit).

        An empty or ``None`` ``revision`` means no revision was requested and is
        always treated as a match.

        :param revision: branch, commit, or tag to compare against HEAD
        :returns: True when the repository is at ``revision`` (or no revision
                  was requested)
        """

        if revision is None or revision.strip() == '':
            # No revision requested; nothing to check against.
            return True

        revision = revision.strip()
        current_commit = self.commit

        # Direct match on the current branch name.
        if revision == self.branch:
            return True

        # A (possibly partial) commit hash that is a prefix of HEAD's commit.
        # Covers a detached checkout at a specific commit, where there is no
        # branch to match on.
        lowered = revision.lower()
        if (all(ch in '0123456789abcdef' for ch in lowered)
                and len(lowered) >= 4
                and current_commit.lower().startswith(lowered)):
            return True

        # Resolve the revision through git (handles branches, tags, and
        # unambiguous partial commits) and compare to the current commit.
        resolved = self.resolve_commit(revision)
        return resolved is not None and resolved == current_commit

    def checkout(self, revision: str) -> None:
        """Checkout a branch, commit, or tag.

        Commits or tags always have HEAD detached.

        :param revision: branch name, commit hash, or tag to check out
        :raises BanditError: if ``revision`` is not a valid branch, commit, or tag
        """

        res = self._git('checkout', revision, allowed_codes=(0, 1))
        if res.returncode == 1:
            raise BanditError(f'Invalid branch, commit, or tag: {revision}')
