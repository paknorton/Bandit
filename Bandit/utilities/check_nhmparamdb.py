#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)

# import argparse
import os
# import shutil

import Bandit.bandit_cfg as bc
from Bandit.git_version import git_version

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    # parser = argparse.ArgumentParser(description='Check the current revision of the local NHMparamdb against GIT')
    # parser.add_argument('-u', '--update', help='Name of new job directory', store_action=True)
    #
    # args = parser.parse_args()

    config = bc.Cfg('bandit.cfg')

    # Get NHMparamdb version that is currently used for the merged parameter database
    with open('{}/00-REVISION'.format(config.merged_paramdb_dir), 'r') as fhdl:
        m_rev = fhdl.readline().strip()

    # Get the current available revision for the NhmParamDb
    c_rev = git_version(config.paramdb_dir)

    if m_rev != c_rev:
        print('A newer version of the NhmParamDb is available.')

        print('  NhmParamDb revision used by bandit: {}'.format(m_rev))
        print('NhmParamDb revision available on GIT: {}'.format(c_rev))
        print('\nTo update the NhmParamDb first change into directory: {}'.format(config.paramdb_dir))
        print("Then type 'git pull'")
        print('After the update is completed change into directory: {}'.format(os.getcwd()))
        print("Then type 'create_merged_db'")
    else:
        print('NhmParamDb is up-to-date.')


if __name__ == '__main__':
    main()
