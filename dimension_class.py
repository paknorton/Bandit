#!/usr/bin/env python2.7

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
from future.utils import iteritems

import numpy as np
import xml.etree.ElementTree as xmlET

from collections import OrderedDict


def read_xml(filename):
    # Open and parse an xml file and return the root of the tree
    xml_tree = xmlET.parse(filename)
    return xml_tree.getroot()


class Dimension(object):
    """Defines a single dimension"""
    # Container for a single dimension
    def __init__(self, position=0, size=0):
        self.__position = 0  # This will get set by self.position below
        self.position = position  # integer
        self.__size = size  # integer

    @property
    def position(self):
        """Returns the ordering of the dimension"""
        # 2016-12-23 PAN: Does this make better sense as the Dimensions level?
        return self.__position

    @position.setter
    def position(self, value):
        """Set the ordering of the dimension"""
        if not isinstance(value, int):
            raise TypeError('Only integers are allowed for dimension position')
        if value < 1 or value > 2:
            # Max # of dimensions for parameters will effect the number of positions available
            raise ValueError('Dimension position must be 1 or 2')
        self.__position = value

    @property
    def size(self):
        """"Return the size of the dimension"""
        return self.__size

    @size.setter
    def size(self, value):
        """Set the size of the dimension"""
        if not isinstance(value, int) or value < 0:
            raise ValueError('Dimension size must be a positive integer')
        self.__size = value

    def __repr__(self):
        return 'Dimension(position={!r}, size={!r})'.format(self.position, self.size)

    def __iadd__(self, other):
        # augment in-place addition so the instance plus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        self.__size += other
        return self

    def __isub__(self, other):
        # augment in-place addition so the instance minus a number results
        # in a change to self.__size
        if not isinstance(other, int):
            raise ValueError('Dimension size type must be an integer')
        if self.__size - other < 0:
            raise ValueError('Dimension size must be positive')
        self.__size -= other
        return self


class Dimensions(object):
    """Container of Dimension objects"""
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
    def dimensions(self):
        """Returns ordered dictionary of Dimension objects"""
        # Return the ordered dictionary of define dimensions
        return self.__dimensions

    @property
    def ndims(self):
        # Number of dimensions
        return len(self.__dimensions)

    def add_dimension(self, name, position=0, size=0, first_set_size=False):
        # This method either adds a dimension if it doesn't exist
        # or increases the size of an existing dimension.
        # The first_set_size argument controls whether the first dimension is only initialized to 0 (False)
        # or allowed to be set/updated with the size argument (True).

        # TODO: Check if position is already in use for new dimensions
        if name not in self.__dimensions:
            if name in ['nmonths', 'ndays', 'one']:
                # Non-additive (static) dimensions; once set don't change
                self.__dimensions[name] = Dimension(position=position, size=size)
            else:
                if position == 1:
                    # Position one can be either set to the size value or set to zero
                    # depending on first_set_size
                    self.__dimensions[name] = Dimension(position=position, size=size * first_set_size)
                else:
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
                self.__dimensions[name] += (size*first_set_size)

    def get_dim_by_position(self, position):
        # Returns the size of the dimension in the given position
        # Raises an error if the position doesn't exist
        for kk, vv in iteritems(self.__dimensions):
            if vv.position == position:
                return vv.size
        raise ValueError('Parameter has no dimension in position {}'.format(position))

    def tostructure(self):
        """Returns a data structure of Dimensions data for serialization"""
        # Return the dimensions info/data as a data structure
        dims = {}
        for kk, vv in iteritems(self.dimensions):
            dims[kk] = {'position': vv.position, 'size': vv.size}
        return dims


class Parameter(object):

    """Container for a single Parameter object"""

    # Container for a single parameter
    def __init__(self, name=None, datatype=None, units=None):
        """Initialize the Parameter object"""
        self.__valid_datatypes = {'I': 1, 'F': 2, 'D': 3, 'S': 4}

        self.__name = name  # string
        self.__datatype = datatype  # ??
        self.__units = units    # string (optional)
        self.__dimensions = Dimensions()
        self.__data = None  # array

    def __str__(self):
        """Return a pretty print string representation of the parameter information"""
        outstr = 'name: {}\ndatatype: {}\nunits: {}\nndims: {}\n'.format(self.name, self.datatype,
                                                                         self.units, self.ndims)
        if self.ndims:
            outstr += 'Dimensions:\n' + self.dimensions.__str__()
        return outstr

    @property
    def name(self):
        """Return the name of the parameter"""
        return self.__name

    @property
    def datatype(self):
        """Return the datatype of the parameter"""
        return self.__datatype

    @property
    def units(self):
        """Return the units used for the parameter data"""
        return self.__units

    @units.setter
    def units(self, unitstr):
        """Set the units used for the parameter data"""
        self.__units = unitstr

    @property
    def dimensions(self):
        """Return the Dimensions object associated with the parameter"""
        return self.__dimensions

    @property
    def ndims(self):
        """Return the number of dimensions that are defined for the parameter"""
        # Return the number of dimensions defined for the parameter
        return self.__dimensions.ndims

    @property
    def data(self):
        """Return the data associated with the parameter"""
        return self.__data

    def add_dimension(self, name, position=0, size=0, first_set_size=False):
        self.__dimensions.add_dimension(name, position=position, size=size, first_set_size=first_set_size)

    def add_dimensions_from_xml(self, filename, first_set_size=False):
        # Add dimensions and grow dimension sizes from xml information for a parameter
        # This information is found in xml files for each region for each parameter
        # No attempt is made to verify whether each region for a given parameter
        # has the same or same number of dimensions.
        xml_root = read_xml(filename)

        for cdim in xml_root.findall('./dimensions/dimension'):
            dim_name = cdim.get('name')
            dim_size = int(cdim.get('size'))

            self.__dimensions.add_dimension(dim_name, position=int(cdim.get('position')), size=dim_size,
                                            first_set_size=first_set_size)

    def append_paramdb_data(self, data):
        # Appends parameter data read from an official parameter database file.
        # This is primarily used when creating a CONUS-level parameter from a set of
        # parameter database regions. Use with caution as no region information is
        # considered, the data is just appended and dimensions are verified.
        # Assume input data is a list of strings

        # Raise and error if no dimensions are defined for parameter
        if not self.ndims:
            raise ValueError('No dimensions have been defined for {}. Unable to append data'.format(self.name))

        # Convert datatype first
        datatype_conv = {'I': self.__str_to_int, 'F': self.__str_to_float,
                         'D': self.__str_to_float, 'S': self.__str_to_str}

        if self.__datatype in self.__valid_datatypes.keys():
            data = datatype_conv[self.__datatype](data)
        else:
            raise TypeError('Defined datatype {} for parameter {} is not valid'.format(self.__datatype,
                                                                                       self.__name))

        # Convert list to np.array
        if self.ndims == 2:
            data_np = np.array(data).reshape((-1, self.__dimensions.get_dim_by_position(2),), order='F')
        elif self.ndims == 1:
            data_np = np.array(data)
        else:
            raise ValueError('Number of dimensions, {}, is not supported'.format(self.ndims))

        # Create/append to internal data structure
        if self.__data is None:
            self.__data = data_np
        else:
            # Data structure has already been created; try concatenating the incoming data
            self.__data = np.concatenate((self.__data, data_np))

        # Resize dimensions to reflect new size
        self.__resize_dims()

    def tolist(self):
        # Return a list of the data
        return self.__data.ravel(order='F').tolist()

    def tostructure(self):
        # Return all information about this parameter in the following form
        param = {}
        param['name'] = self.name
        param['datatype'] = self.datatype
        param['dimensions'] = self.dimensions.tostructure()
        param['data'] = self.tolist()
        return param

    def __resize_dims(self):
        # Adjust dimension size(s) to reflect the data
        if self.ndims == 2:
            # At least for now only the 1st dimension can grow
            self.__dimensions.dimensions.values()[0].size = self.__data.shape[0]
        elif self.ndims == 1:
            self.__dimensions.dimensions.values()[0].size = self.__data.shape[0]

    def __str_to_float(self, data):
        # Convert provide list of data to float
        try:
            return [float(vv) for vv in data]
        except ValueError as ve:
            print(ve)

    def __str_to_int(self, data):
        # Convert list of data to integer
        try:
            return [int(vv) for vv in data]
        except ValueError as ve:
            print(ve)

    def __str_to_str(self, data):
        # nop for list of strings
        return data


class Parameters(object):
    # Container for a collection of Parameter objects
    def __init__(self):
        self.__parameters = None  # array? dict? of Parameter()




