#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)
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
    paramdb_dir = '/Users/pnorton/Projects/National_Hydrology_Model/paramDb/nhmparamdb'
    print('GIT version for {}'.format(paramdb_dir))
    print(git_version(paramdb_dir))


if __name__ == '__main__':
    main()
