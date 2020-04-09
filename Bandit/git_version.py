#!/usr/bin/env python3

# from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems, iterkeys

import os
import subprocess


def git_version(repo_dir):
    """Retrieve current git version from local directory.

    Args:
        repo_dir (str): Local git repository directory.

    Returns:
        str: Current git revision or 'Unknown'.

    """

    # Return the git revision as a string
    # Modified version of function from numpy setup.py
    # http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    # Accessed on 2017-02-24
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}

        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v

        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return result

    try:
        out = _minimal_ext_cmd(['git', '-C', repo_dir, 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def main():
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser(description='Get current revision of GIT repo')
    parser.add_argument('-p', '--path_repo', help='Path to local repository')
    parser.add_argument('--verbose', help='Verbose output', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print('GIT version for {}'.format(args.path_repo))

    print(git_version(args.path_repo))


if __name__ == '__main__':
    main()
