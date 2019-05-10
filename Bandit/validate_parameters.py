#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function)

import argparse
import os
from pyPRMS.ControlFile import ControlFile
import pyPRMS.ValidParams as vparm
import pyPRMS.NhmParamDb as nhm
import pyPRMS.ParameterFile as pfile

__author__ = 'Parker Norton (pnorton@usgs.gov)'


def main():
    parser = argparse.ArgumentParser(description='Validate parameter file or paramdb parameters')
    parser.add_argument('-c', '--control', required=True, help='Name of control file', nargs='?', default='control.default', type=str)
    parser.add_argument('-x', '--xml', help='Parameter xml file', type=str)
    # parser.add_argument('-x', '--xml', help='Parameter xml file', nargs='?', default='parameters.xml', type=str)
    parser.add_argument('-p', '--param_src', required=True, help='Parameter file or paramdb')

    args = parser.parse_args()

    if os.path.splitext(args.param_src)[1] == '.param':
        print('Reading {}'.format(args.param_src))
        pdb = pfile.ParameterFile(args.param_src)
    elif os.path.isdir(args.param_src):
        print('Reading {}'.format(args.param_src))
        pdb = nhm.NhmParamDb(args.param_src)
    else:
        print('ERROR: No valid parameter set specified')
        exit(1)

    print('Reading {}'.format(args.control))
    ctl = ControlFile(args.control)

    modules_used = []

    for xx in ctl.modules.keys():
        if xx == 'precip_module':
            if ctl.modules[xx] == 'climate_hru':
                modules_used.append('precipitation_hru')
        elif xx == 'temp_module':
            if ctl.modules[xx] == 'climate_hru':
                modules_used.append('temperature_hru')
        else:
            modules_used.append(ctl.modules[xx])

    # Read the full set of possible parameters
    if args.xml:
        print('Reading {}'.format(args.xml))
        vpdb = vparm.ValidParams(args.xml)
    else:
        print('Reading default parameters.xml')
        vpdb = vparm.ValidParams()

    print()
    # Build dictionary of parameters by module from the set of master parameters
    params_by_module = {}

    for xx in vpdb.parameters.values():
        for mm in xx.modules:
            if mm not in params_by_module:
                params_by_module[mm] = []
            params_by_module[mm].append(xx.name)

    for xx in params_by_module.keys():
        if xx in modules_used:
            print(xx)

            # Output parameters that are required by the selected modules
            # but missing from the parameter file or paramdb
            for yy in params_by_module[xx]:
                if not pdb.parameters.exists(yy):
                    if yy in ['basin_solsta', 'hru_solsta', 'rad_conv']:
                        try:
                            if pdb.dimensions.get('nsol') > 0:
                                print('\tMissing: {}'.format(yy))
                        except ValueError:
                            pass
                    elif yy == 'irr_type':
                        try:
                            if pdb.dimensions.get('nwateruse') == 1:
                                print('\tMissing: {}'.format(yy))
                        except ValueError:
                            pass
                    elif yy == 'gvr_hru_id':
                        try:
                            if ctl.get_var('mapOutON_OFF') and ctl.get_values('mapOutON_OFF') == 1:
                                print('\tMissing: {}'.format(yy))
                        except ValueError:
                            pass
                    else:
                        print('\tMissing: {}'.format(yy))


if __name__ == '__main__':
    main()