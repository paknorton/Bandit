

from __future__ import (absolute_import, division, print_function)
# , unicode_literals)
# from future.utils import iteritems

from collections import OrderedDict
from osgeo import ogr
import gdal


class GdalErrorHandler(object):

    """Error handling class for GDAL."""

    # See: https://trac.osgeo.org/gdal/wiki/PythonGotchas
    # We define this class and add error handling code to class Geo
    # so that warnings can be intercepted by Python's exception handling
    def __init__(self):
        """Create the GdalErrorHandler object."""

        self.err_level = gdal.CE_None
        self.err_no = 0
        self.err_msg = ''

    def handler(self, err_level, err_no, err_msg):
        """Set error message information.

        :param int err_level: error level
        :param int err_no: error number
        :param str err_msg: error message
        """

        self.err_level = err_level
        self.err_no = err_no
        self.err_msg = err_msg


class Geo(object):

    """Class for subsetting GIS files."""

    def __init__(self, filename):
        """Create the Geo object.

        :param str filename: name of file
        """

        err = GdalErrorHandler()
        handler = err.handler

        gdal.PushErrorHandler(handler)
        gdal.UseExceptions()

        self.__filename = filename
        self.__layers = None
        self.__selected_layer = None

        # use OGR specific exceptions
        ogr.UseExceptions()

        # Load the file - this assumes a file geodatabase
        driver = ogr.GetDriverByName('OpenFileGDB')
        self.__gdb = driver.Open(self.__filename)

    @property
    def layers(self):
        """Get dictionary mapping layer names to their index.

        :returns: dictionary of layer name to index value
        :rtype: dict[str, int]
        """

        # Returns a dictionary mapping layer names to their index
        if not self.__layers:
            self.__layers = OrderedDict()

            for lyr_idx in range(self.__gdb.GetLayerCount()):
                lyr = self.__gdb.GetLayerByIndex(lyr_idx)
                self.__layers[lyr.GetName()] = lyr_idx
        return self.__layers

    @property
    def selected_layer(self):
        """Get the currently selected layer.

        :returns: selected layer
        :rtype: int or None
        """

        return self.__selected_layer

    def select_layer(self, layer_name):
        """Set the selected layer.

        :param str layer_name: name of layer to select
        """

        # Select a layer from the file geodatabase
        self.__selected_layer = self.__gdb.GetLayerByName(layer_name)

    def filter_by_attribute(self, attr_name, attr_values):
        """Filter current layer by attribute name and values.

        :param str attr_name: name of attribute
        :param list attr_values: list of attribute values
        """

        # Filter a layer by attribute name and values

        # Make sure the attr_values elements are strings
        attr_args = ','.join([str(xx) for xx in attr_values])
        print(len(attr_args))
        print('-'*50)
        print(attr_args)
        print('-' * 50)

        self.__selected_layer.SetAttributeFilter('{} in ({})'.format(attr_name, attr_args))

    def write_shapefile(self, filename, attr_name, attr_values, included_fields=None):
        """Write subset to shapefile format.

        :param str filename: name of shapefile to create
        :param str attr_name: name of attribute for filtering
        :param list attr_values: list of attribute values to include in subset
        :param included_fields: list of attribute field names from source shapefile to include in new shapefile
        :type included_fields: None or list[str]
        """

        # Create a shapefile for the current selected layer
        # If a filter is set then a subset of features is written

        limit_fields = included_fields is not None

        out_driver = ogr.GetDriverByName('ESRI Shapefile')

        out_ds = out_driver.CreateDataSource(filename)
        out_layer = out_ds.CreateLayer(self.__selected_layer.GetName(), self.__selected_layer.GetSpatialRef())

        # Copy field definitions from input to output file
        in_layer_def = self.__selected_layer.GetLayerDefn()

        orig_fld_names = []
        for ii in range(in_layer_def.GetFieldCount()):
            fld_def = in_layer_def.GetFieldDefn(ii)
            fld_name = fld_def.GetName()

            if limit_fields and fld_name not in included_fields:
                continue

            orig_fld_names.append(fld_name)
            out_layer.CreateField(fld_def)

        # Add model_idx field
        fld_def = ogr.FieldDefn('model_idx', ogr.OFTInteger)
        orig_fld_names.append('model_idx')
        out_layer.CreateField(fld_def)

        # Get feature definitions for the output layer
        out_layer_def = out_layer.GetLayerDefn()

        # Create blank output feature
        out_feat = ogr.Feature(out_layer_def)

        # Add features to the output layer
        for in_feat in self.__selected_layer:
            if in_feat.GetField(attr_name) in attr_values:
                # Add field values from the input layer
                for ii in range(out_layer_def.GetFieldCount()):
                    fld_def = out_layer_def.GetFieldDefn(ii)
                    fld_name = fld_def.GetName()

                    if fld_name == 'model_idx':
                        # Output a 1-based array index value
                        out_feat.SetField(out_layer_def.GetFieldDefn(ii).GetNameRef(), attr_values.index(in_feat.GetField(attr_name))+1)
                    else:
                        out_feat.SetField(out_layer_def.GetFieldDefn(ii).GetNameRef(), in_feat.GetField(orig_fld_names[ii]))

                # Set geometry as centroid
                # geom = in_feat.GetGeometryRef
                # out_feat.SetGeometry(geom.Clone())

                # Set geometry
                geom = in_feat.geometry()
                out_feat.SetGeometry(geom)

                # Add the new feature to the output layer
                out_layer.CreateFeature(out_feat)

        # Close the output datasource
        out_ds.Destroy()

    def write_shapefile2(self, filename):
        # Create a shapefile for the current selected layer
        # Any applied filter will effect what is written to the new file
        # TODO raise error if no layer is selected
        out_driver = ogr.GetDriverByName('ESRI Shapefile')
        out_ds = out_driver.CreateDataSource(filename)

        out_ds.CopyLayer(self.__selected_layer, self.__selected_layer.GetName())
        del out_ds

    def write_shapefile3(self, filename, attr_name, attr_values, included_fields=None):
        # NOTE: This is for messing around with the geopackage format

        # Create a shapefile for the current selected layer
        # If a filter is set then a subset of features is written
        print(attr_values)
        out_driver = ogr.GetDriverByName('GPKG')

        out_ds = out_driver.CreateDataSource('crap.gpkg')
        # out_ds = out_driver.CreateDataSource(filename)
        out_layer = out_ds.CreateLayer(self.__selected_layer.GetName(), self.__selected_layer.GetSpatialRef())

        # Copy field definitions from input to output file
        in_layer_def = self.__selected_layer.GetLayerDefn()

        for ii in range(in_layer_def.GetFieldCount()):
            fld_def = in_layer_def.GetFieldDefn(ii)
            fld_name = fld_def.GetName()

            if included_fields and fld_name not in included_fields:
                continue
            out_layer.CreateField(fld_def)

        # Get feature definitions for the output layer
        out_layer_def = out_layer.GetLayerDefn()

        # Create blank output feature
        out_feat = ogr.Feature(out_layer_def)

        # Add features to the output layer
        for in_feat in self.__selected_layer:
            if in_feat.GetField(attr_name) in attr_values:
                print(in_feat.GetField(attr_name))
                # Add field values from the input layer
                for ii in range(out_layer_def.GetFieldCount()):
                    fld_def = out_layer_def.GetFieldDefn(ii)
                    fld_name = fld_def.GetName()

                    if included_fields and fld_name not in included_fields:
                        continue
                    out_feat.SetField(out_layer_def.GetFieldDefn(ii).GetNameRef(), in_feat.GetField(ii))

                # Set geometry as centroid
                # geom = in_feat.GetGeometryRef()
                # out_feat.SetGeometry(geom.Clone())

                # Set geometry
                geom = in_feat.geometry()
                out_feat.SetGeometry(geom)

                # Add the new feature to the output layer
                out_layer.CreateFeature(out_feat)

        # Close the output datasource
        out_ds.Destroy()

    def write_kml(self, filename):
        # Create a shapefile for the current selected layer
        # Any applied filter will effect what is written to the new file
        out_driver = ogr.GetDriverByName('KML')
        out_ds = out_driver.CreateDataSource(filename)

        out_ds.CopyLayer(self.__selected_layer, self.__selected_layer.GetName())
        del out_ds
