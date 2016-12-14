
import pandas as pd

VALID_VARNAMES = ['prcp', 'tmin', 'tmax']


class Cbh(object):
    # Author: Parker Norton (pnorton@usgs.gov)
    # Create date: 2016-12-05
    # Description: Class for working with individual cbh files
    #
    # This class assumes it is dealing with regional cbh files (not a CONUS-level NHM file)
    # TODO: As written type of data (e.g. tmax, tmin, prcp) is ignored.
    # TODO: Verify that given data type size matches number of columns

    def __init__(self, filename, indices=None):
        self.__filename = filename
        self.__indices = indices    # This should be an ordered dict: nhm->local hrus
        self.__data = None

    @property
    def data(self):
        return self.__data

    def read_cbh(self):
        # in_hdl = open(self.__filename, 'r')
        # fheader = ''
        #
        # # Read the header information
        # for ii in range(0, 3):
        #     line = in_hdl.readline()
        #
        #     if line[0:4] in VALID_VARNAMES:
        #         # Not needed right now; has been used to subset the total number of values for output
        #         fheader += line
        #         # fheader += line[0:5] + ' 1\n'
        #     else:
        #         fheader += line

        # Read the data
        if self.__indices:
            incl_cols = [0, 1, 2, 3, 4, 5]
            for xx in self.__indices.values():
                incl_cols.append(xx+5)  # include an offset for the having datetime info

            # Columns 0-5 always represent date/time information
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True, usecols=incl_cols,
                                      skiprows=3, engine='c', header=None)

            # Rename columns with NHM HRU ids
            ren_dict = {v+5: k for k, v in self.__indices.iteritems()}
            # print(ren_dict)

            # NOTE: The rename is an expensive operation
            self.__data.rename(columns=ren_dict, inplace=True)
        else:
            # Read the entire file
            self.__data = pd.read_csv(self.__filename, sep=' ', skipinitialspace=True,
                                      skiprows=3, engine='c', header=None)
        # in_hdl.close()

