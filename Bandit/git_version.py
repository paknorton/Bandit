#!/usr/bin/env python3

import os
import subprocess

from pathlib import Path
from typing import List, Optional, Union

from pyPRMS.base.console import get_console_instance

con = None

# Modified version of function from numpy setup.py
# http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
# Accessed on 2017-02-24

class Repo:
    def __init__(self, path: str | Path):
        global con
        con = get_console_instance()

        if isinstance(path, str):
            path = Path(path)
        self.path_repo = path.resolve()

    def _git(self, *args, allowed_codes=(0,)):
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
    def branch(self) -> str:
        """Get the current branch.

        :returns: current branch for local repository
        """

        result = self._git('symbolic-ref', '--quiet', '--short', 'HEAD', allowed_codes=(0, 1))
        return result.stdout.rstrip() if result.returncode == 0 else None

    @property
    def commit(self) -> str:
        """Retrieve current commit number.
        """

        return self._git('rev-parse', '--verify', 'HEAD').stdout.strip()

    @property
    def remote_url(self) -> str | None:
        """Returns the remote commit URL for the branch or commit

        This always returns a URL for commit whether detached or associated with a branch.
        If it's a detached commit remote origin is assumed.
        """

        branch = self.branch
        if branch is None:
            result = self._git('config', '--get', 'remote.origin.url').stdout.rstrip()
        else:
            br_remote = self._git('config', '--get', f'branch.{branch}.remote').stdout.rstrip()

            # remote = br_remote.stdout.rstrip()
            if br_remote == '.':
                return None  # Upstream is another branch in this repository.
            result = self._git('remote', 'get-url', br_remote).stdout.rstrip()

        final_url = result.replace(':', '/').replace('.git', '').replace('git@', 'https://')
        final_url += f'/-/commit/{self.commit}'

        return final_url

    def checkout(self, bct: str):
        """Checkout a branch, commit, or tag

        Commits or tags always have HEAD detached
        """
        res = self._git('checkout', bct, allowed_codes=(0, 1))
        if res.returncode == 1:
            con.print('[red]ERROR[/]: Invalid branch, commit, or tag')

