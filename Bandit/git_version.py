#!/usr/bin/env python3

# from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems, iterkeys

import os
import subprocess


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


def git_commit_url(repo_dir):
    src_url = git_repo(repo_dir)
    src_commit = git_commit(repo_dir)

    dst_url = src_url.replace(':', '/').replace('.git', '').replace('git@', 'https://')
    dst_url += f'/commit/{src_commit}'
    return dst_url


def git_commit(repo_dir, length=None):
    """Retrieve current git version from local directory.

    Args:
        repo_dir (str): Local git repository directory.
        length (int): Number of digits to return from commit

    Returns:
        str: Current git revision or 'Unknown'.

    """

    # Return the git revision as a string
    try:
        if length is None:
            out = _minimal_ext_cmd(['git', '-C', repo_dir, 'rev-parse', 'HEAD'])
        else:
            out = _minimal_ext_cmd(['git', '-C', repo_dir, 'rev-parse', f'--short={length}', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def git_repo(repo_dir):
    # Return the git repo URL as a string
    try:
        out = _minimal_ext_cmd(['git', '-C', repo_dir, 'config', '--get', 'remote.origin.url'])
        git_repo_url = out.strip().decode('ascii')
    except OSError:
        git_repo_url = "Unknown"

    return git_repo_url


def git_branch(repo_dir):
    # Return the git branch as a string
    try:
        out = _minimal_ext_cmd(['git', '-C', repo_dir, 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_repo_branch = out.strip().decode('ascii')
    except OSError:
        git_repo_branch = "Unknown"

    return git_repo_branch


def main():
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser(description='Get current revision of GIT repo')
    parser.add_argument('-p', '--path_repo', help='Path to local repository')
    parser.add_argument('--verbose', help='Verbose output', action='store_true')

    args = parser.parse_args()

    if args.verbose:
        print(f'GIT version for {args.path_repo}')

    print(f'GIT repo URL: {git_repo(args.path_repo)}')
    print(f'GIT branch: {git_branch(args.path_repo)}')
    print(f'GIT commit: {git_commit(args.path_repo)}')


if __name__ == '__main__':
    main()
