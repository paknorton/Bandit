#!/usr/bin/env python3

# from __future__ import (absolute_import, division, print_function)
# from future.utils import iteritems

import Bandit.colortest as color


def print_warning(thestr):
    colorize('WARNING: ', 'yellow')
    print(thestr)


def print_info(thestr):
    colorize('INFO: ', 'green')
    print(thestr)


def print_error(thestr):
    colorize('ERROR: ', 'red')
    print(thestr)


def div(thestr, padchar='*', width=80):
    lpad = int((width - len(thestr))/2)
    rpad = int(width - lpad - len(thestr))
    return padchar * lpad + ' ' + thestr + ' ' + padchar * rpad


def heading(thestr, tag=None):
    print('')
    color.term(tag + ' ' + div(thestr), end='')
    color.term(end='\n')
    print('')


def printlist(thelist, title):
    # Given a list, print it out
    print('**** %s ****' % title)
    for x in thelist:
        print('\t%s' % str(x))
        
    print('Items in list: %d\n' % (len(thelist)))


def show_dict(thedict, title):
    color.term('white green' + ' ' + div(title, '-', 40), end='')
    color.term(end='\n')
    print('')
    for i, j in thedict.items():
        show_var(i, str(j), 'green', '')
    color.term('white green' + ' ' + div('', '-', 40), end='')
    color.term(end='\n')
    print('')


def show_var(thevar, theval, vartag, valtag, sep='\t'):
    color.term(vartag + ' ' + thevar, end='')
    color.term(valtag + ' ' + sep + theval, end='')
    color.term()
    print('')


def colorize(thestr, tag):
    tmp = '%s %s' % (tag, thestr)
    color.term(tmp, end='')
    color.term(end='')
