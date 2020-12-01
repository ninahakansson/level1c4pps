#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 level1c4pps developers
#
# This file is part of level1c4pps
#
# level1c4pps is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# level1c4pps is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with level1c4pps.  If not, see <http://www.gnu.org/licenses/>.
# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Nina Hakansson <nina.hakansson@smhi.se>
#   Adam.Dybbroe <adam.dybbroe@smhi.se>

"""Utilities to convert AVHRR GAC formattet data to PPS level-1c format."""


import os
import time
from satpy.scene import Scene
from level1c4pps import (get_encoding, compose_filename,
                         set_header_and_band_attrs_defaults,
                         remove_attributes,
                         rename_latitude_longitude, update_angle_attributes,
                         platform_name_to_use_in_filename,
                         fix_too_great_attributes,
                         get_header_attrs, convert_angles)
import logging
from satpy.utils import debug_on
from distutils.version import LooseVersion
import satpy
if LooseVersion(satpy.__version__) < LooseVersion('0.24.0'):
    debug_on()
    raise ImportError("'eumgac2pps' writer requires satpy 0.24.0 or greater")


# AVHRR-GAC_FDR_1C_N06_19810330T005421Z_19810330T024632Z_R_O_20200101T000000Z_0100.nc

logger = logging.getLogger('eumgacfdr2pps')

BANDNAMES = ['reflectance_channel_1',
             'reflectance_channel_2',
             'reflectance_channel_3',
             'brightness_temperature_channel_3',
             'brightness_temperature_channel_4',
             'brightness_temperature_channel_5']

ANGLENAMES = ['sensor_zenith_angle',
              'solar_zenith_angle',
              'solar_azimuth_angle',
              'sensor_azimuth_angle',
              'sun_sensor_azimuth_difference_angle']

REFL_BANDS = ['reflectance_channel_1', 'reflectance_channel_2', 'reflectance_channel_3']

PPS_TAGNAMES = {"reflectance_channel_1": "ch_r06",
                "reflectance_channel_2": "ch_r09",
                "reflectance_channel_3": "ch_r16",
                "brightness_temperature_channel_3": "ch_tb37",
                "brightness_temperature_channel_4": "ch_tb11",
                "brightness_temperature_channel_5": "ch_tb12"}


RENAME_AND_MOVE_TO_HEADER = {'id': 'euemtsat_gac_id',
                             'licence': 'eumetsat_licence',
                             'product_version': 'eumetsat_product_version',
                             'version_satpy': 'version_eumetsat_pygac_fdr_satpy'}


def get_encoding_gac(scene):
    """Get netcdf encoding for all datasets."""
    return get_encoding(scene,
                        BANDNAMES,
                        PPS_TAGNAMES,
                        chunks=None)


def update_ancilliary_datasets(scene):
    """Rename, delete and add some datasets and attributes."""
    irch = scene['brightness_temperature_channel_4']

    # Create new data set scanline timestamps
    scene['scanline_timestamps'] = scene['acq_time']
    scene['scanline_timestamps'].attrs['name'] = 'scanline_timestamps'
    del scene['acq_time'].coords['acq_time']
    del scene['acq_time']

    # Update qual_flags attrs
    scene['qual_flags'].attrs['id_tag'] = 'qual_flags'
    scene['qual_flags'].attrs['long_name'] = 'pygac quality flags'
    scene['qual_flags'].coords['time'] = irch.attrs['start_time']
    del scene['qual_flags'].coords['acq_time']
    for band in ['scanline_timestamps',
                 'qual_flags',
                 'overlap_free_end',
                 'overlap_free_end',
                 'equator_crossing_time',
                 'equator_crossing_longitude',
                 'midnight_line']:
        scene[band].encoding.pop('coordinates', None)
        for attr in scene[band].attrs:
            if attr not in ['_FillValue', 'long_name', 'standard_name', 'units']:
                scene[band].attrs.pop('coordinates', None)


def set_header_and_band_attrs(scene):
    """Set and delete some attributes."""
    irch = scene['brightness_temperature_channel_4']
    for attr in RENAME_AND_MOVE_TO_HEADER:
        if attr in irch.attrs:
            scene.attrs[RENAME_AND_MOVE_TO_HEADER[attr]] = irch.attrs.pop(attr)
    nimg = set_header_and_band_attrs_defaults(scene, BANDNAMES, PPS_TAGNAMES, REFL_BANDS, irch)
    scene.attrs['source'] = "eumgacfdr2pps.py"
    scene.attrs['is_gac'] = 'True'
    scene.attrs['orbit_number'] = 99999
    for band in BANDNAMES:
        if band not in scene:
            continue
        if band in REFL_BANDS:
            scene[band].attrs['sun_earth_distance_correction_applied'] = 'True'
        del scene[band].encoding['coordinates']
    return nimg


def process_one_file(eumgacfdr_file, out_path='.', reader_kwargs=None, engine='h5netcdf'):
    """Make level 1c files in PPS-format."""
    tic = time.time()
    scn_ = Scene(reader='avhrr_l1c_eum_gac_fdr_nc',
                 filenames=[eumgacfdr_file])

    scn_.load(BANDNAMES)
    scn_.load(['latitude',
               'longitude',
               'qual_flags',
               'acq_time',
               'overlap_free_start',
               'overlap_free_end',
               'equator_crossing_time',
               'equator_crossing_longitude',
               'midnight_line'] +
              ANGLENAMES)

    # One ir channel
    irch = scn_['brightness_temperature_channel_4']

    # Set header and band attributes
    set_header_and_band_attrs(scn_)

    # Rename longitude, latitude to lon, lat.
    rename_latitude_longitude(scn_)

    # Convert angles to PPS
    convert_angles(scn_)
    update_angle_attributes(scn_, irch)  # Standard name etc

    # Handle gac specific datasets qual_flags and scanline_timestamps
    update_ancilliary_datasets(scn_)

    filename = compose_filename(scn_, out_path, instrument='avhrr', band=irch)
    encoding = get_encoding_gac(scn_)
    scn_.save_datasets(writer='cf',
                       filename=filename,
                       header_attrs=get_header_attrs(scn_, band=irch, sensor='avhrr'),
                       engine=engine,
                       flatten_attrs=True,
                       include_lonlats=False,  # Included anyway as they are datasets in scn_
                       pretty=True,
                       encoding=encoding)

    print("Saved file {:s} after {:3.1f} seconds".format(
        os.path.basename(filename),
        time.time()-tic))
    return filename
