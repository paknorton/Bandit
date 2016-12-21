#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

import xml.etree.ElementTree as xmlET

from collections import OrderedDict


def read_xml(filename):
    # Open and parse an xml file and return the root of the tree
    xml_tree = xmlET.parse(filename)
    return xml_tree.getroot()


class Dimension(object):
    # Container for a single dimension
    def __init__(self, position=None, size=0):
        self.__position = position  # integer
        self.__size = size  # integer

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, value):
        if not isinstance(value, int):
            raise TypeError
        if value < 0 or value > 3:
            # Max # of dimensions for parameters will effect the number of positions available
            raise ValueError
        self.__position = value

    @property
    def size(self):
        return self.__size

    def __repr__(self):
        return 'Dimension(position={!r}, size={!r})'.format(self.__position, self.__size)

    def __iadd__(self, other):
        # augment in-place addition so the instance plus a number results
        # in a change to self.__size
        self.__size += other
        return self

    def __isub__(self, other):
        # augment in-place addition so the instance minus a number results
        # in a change to self.__size
        if self.__size - other < 0:
            raise ValueError
        self.__size -= other
        return self


class Dimensions(object):
    # Container for a collection of dimensions
    def __init__(self):
        self.__dimensions = OrderedDict()  # array? dict? of Dimension()

    def __str__(self):
        outstr = ''
        for kk, vv in iteritems(self.__dimensions):
            outstr += '{}: {}\n'.format(kk, vv)
        return outstr

    def __getattr__(self, attrname):
        if attrname not in self.__dimensions:
            raise AttributeError('Dimension {} does not exist'.format(attrname))
        return self.__dimensions[attrname]

    @property
    def ndims(self):
        # Number of dimensions
        return len(self.__dimensions)

    def add_dimension(self, name, position=0, size=0):
        # This method either adds a dimension if it doesn't exist
        # or increases the size of an existing dimension.
        if name not in self.__dimensions:
            self.__dimensions[name] = Dimension(position=position, size=size)
        else:
            # dimension already exists
            if position != self.__dimensions[name].position:
                # This indicates a problem in one of the paramdb files
                raise ValueError('{}: Attempted position change from {} to {}'.format(name,
                                                                             self.__dimensions[name].position,
                                                                             position))
            if name not in ['nmonths', 'ndays', 'one']:
                # Only add new size for additive dimensions
                self.__dimensions[name] += size


class Parameter(object):
    # Container for a single parameter
    def __init__(self, name=None, datatype=None, units=None):
        self.__valid_datypes = {'I': 1, 'F': 2, 'D': 3, 'S': 4}

        self.__name = name  # string
        self.__datatype = datatype  # ??
        self.__units = units    # string (optional)
        self.__dimensions = Dimensions()
        self.__data = None  # array

    @property
    def name(self):
        return self.__name

    @property
    def datatype(self):
        return self.__datatype

    @property
    def units(self):
        return self.__units

    @units.setter
    def units(self, unitstr):
        self.__units = unitstr

    def add_dimensions_from_xml(self, filename):
        # Add dimensions and grow dimension sizes from xml information for each parameter in each region
        # No attempt is made to verify whether each region for a given parameter
        # have the same or same number of dimensions.
        xml_root = read_xml(filename)

        for cdim in xml_root.findall('./dimensions/dimension'):
            dim_name = cdim.get('name')
            dim_size = int(cdim.get('size'))

            self.__dimensions.add_dimension(dim_name, position=int(cdim.get('position')), size=dim_size)


class Parameters(object):
    # Container for a collection of Parameter objects
    def __init__(self):
        self.__parameters = None  # array? dict? of Parameter()




