#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2010 Stefano Mazzucco
# Copyright 2011 The Hyperspy developers
#
# This file is part of  Hyperspy. It is a fork of the original PIL dm3 plugin
# written by Stefano Mazzucco.
#
#  Hyperspy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  Hyperspy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with  Hyperspy.  If not, see <http://www.gnu.org/licenses/>.

# Plugin to read the Gatan Digital Micrograph(TM) file format

import argparse
import os
import struct
from io import IOBase

import numpy

# Plugin characteristics
# ----------------------
format_name = "Digital Micrograph dm3"
description = "Read data from Gatan Digital Micrograph (TM) files"
full_suport = False
# Recognised file extension
file_extensions = ("dm3", "DM3", "dm4", "DM4")
default_extension = 0

# Writing features
writes = False


def read_short(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(2)  # hexadecimal representation
        if endian == "big":
            s = B_short
        elif endian == "little":
            s = L_short
        return s.unpack(data)[0]  # struct.unpack returns a tuple


def read_ushort(f, endian):
    """Read a 2-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(2)
        if endian == "big":
            s = B_ushort
        elif endian == "little":
            s = L_ushort
        return s.unpack(data)[0]


def read_long(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == "big":
            s = B_long
        elif endian == "little":
            s = L_long
        return s.unpack(data)[0]


def read_longlong(f, endian):
    """Read a 8-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(8)
        if endian == "big":
            s = B_longlong
        elif endian == "little":
            s = L_longlong
        return s.unpack(data)[0]


def read_ulong(f, endian):
    """Read a 4-Byte integer from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == "big":
            s = B_ulong
        elif endian == "little":
            s = L_ulong
        return s.unpack(data)[0]


def read_float(f, endian):
    """Read a 4-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(4)
        if endian == "big":
            s = B_float
        elif endian == "little":
            s = L_float
        return s.unpack(data)[0]


def read_double(f, endian):
    """Read a 8-Byte floating point from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(8)
        if endian == "big":
            s = B_double
        elif endian == "little":
            s = L_double
        return s.unpack(data)[0]


def read_boolean(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == "big":
            s = B_bool
        elif endian == "little":
            s = L_bool
        return s.unpack(data)[0]


def read_byte(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == "big":
            s = B_byte
        elif endian == "little":
            s = L_byte
        return s.unpack(data)[0]


def read_char(f, endian):
    """Read a 1-Byte charater from file f
    with a given endianness (byte order).
    endian can be either 'big' or 'little'.
    """
    if (endian != "little") and (endian != "big"):
        print(("File address:", f.tell()))
        raise ByteOrderError(endian)
    else:
        data = f.read(1)
        if endian == "big":
            s = B_char
        elif endian == "little":
            s = L_char
        return s.unpack(data)[0]


B_short = struct.Struct(">h")
L_short = struct.Struct("<h")

B_ushort = struct.Struct(">H")
L_ushort = struct.Struct("<H")

B_long = struct.Struct(">l")
L_long = struct.Struct("<l")

B_ulong = struct.Struct(">L")
L_ulong = struct.Struct("<L")

B_longlong = struct.Struct(">q")
L_longlong = struct.Struct("<q")

B_float = struct.Struct(">f")
L_float = struct.Struct("<f")

B_double = struct.Struct(">d")
L_double = struct.Struct("<d")

B_bool = struct.Struct(">B")  # use unsigned char
L_bool = struct.Struct("<B")

B_byte = struct.Struct(">b")  # use signed char
L_byte = struct.Struct("<b")

B_char = struct.Struct(">c")
L_char = struct.Struct("<c")

# ----------------------


class DigitalMicrographReader(object):
    """ Class to read Gatan Digital Micrograph (TM) files.

    Currently it supports version 4 only.
    
    Attributes
    ----------
    dm_version, endian, tags_dict
    
    Methods
    -------
    parse_file, parse_header, get_image_dictionaries

    """

    _complex_type = (15, 18, 20)
    simple_type = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

    def __init__(self, f, verbose=False):
        self.verbose = verbose
        self.dm_version = None
        self.endian = None
        self.tags_dict = None
        if isinstance(f, IOBase):
            self.f = f
        else:
            self.f = open(f, "rb")

    def parse_file(self):
        self.f.seek(0)
        self.parse_header()
        self.tags_dict = {"root": {}}
        number_of_root_tags = self.parse_tag_group()[2]
        if self.verbose is True:
            print(("Total tags in root group:", number_of_root_tags))
        self.parse_tags(
            number_of_root_tags, group_name="root", group_dict=self.tags_dict
        )

    def get_image_info(self):
        if self.tags_dict == None:
            self.parse_file()
        image_data = self.tags_dict["ImageList"]["TagGroup1"]["ImageData"]
        x = int(image_data["Dimensions"]["Data0"])
        y = int(image_data["Dimensions"]["Data1"])
        if "Data2" in list(image_data["Dimensions"].keys()):
            z = int(image_data["Dimensions"]["Data2"])
        else:
            z = 1
        header_size = int(image_data["Data"]["offset"])

        type_string = self.tags_dict["ImageList"]["TagGroup1"]["ImageData"]["DataType"]
        endian = self.tags_dict["ImageList"]["TagGroup1"]["ImageData"]["Data"]["endian"]
        depth = self.tags_dict["ImageList"]["TagGroup1"]["ImageData"]["PixelDepth"]

        # DM4 format for DataType
        # 0            null
        # 1     i2     2 byte integer signed ("short")
        # 2     f4     4 byte real (IEEE 754)
        # 3     c8     8 byte complex (real, imaginary)
        # 4            obsolete
        # 5     c4     4 byte packed complex (see DM2)
        # 6    ui1     1 byte integer unsigned ("byte")
        # 7     i4     4 byte integer signed ("long")
        # 8  4*ui1     rgb, 4 bytes/pixel, unused, red, green, blue
        # 9     i1     1 byte integer signed
        # 10    ui2     2 byte integer unsigned
        # 11    ui4     4 byte integer unsigned
        # 12     f8     8 byte real
        # 13     c16   16 byte complex
        # 14     i1     1 byte binary (ie 0 or 1)
        # 23  4*ui1     rgba, 4 bytes/pixel, 0, red, green, blue. Used for thumbnail images

        # convert DM4's types to numpy's dtype
        dtyped = {"1": "i2"}  # 2 byte integer signed ("short")
        dtyped["2"] = "f4"  # 4 byte real (IEEE 754)
        dtyped["3"] = "c8"  # 8 byte complex (real, imaginary)
        dtyped["5"] = "c4"  # 4 byte packed complex (see DM2)
        dtyped["6"] = "u1"  # 1 byte integer unsigned ("byte")
        dtyped["7"] = "i4"  # 4 byte integer signed ("long")
        dtyped["8"] = "u4"  # rgb, 4 bytes/pixel, unused, red, green, blue
        dtyped["9"] = "i1"  # 1 byte integer signed
        dtyped["10"] = "u2"  # 2 byte integer unsigned
        dtyped["11"] = "u4"  # 4 byte integer unsigned
        dtyped["12"] = "f8"  # 8 byte real
        dtyped["13"] = "c16"  # 16 byte complex
        dtyped["14"] = "i1"  # 1 byte binary (ie 0 or 1)
        dtyped[
            "23"
        ] = "u4"  # rgba, 4 bytes/pixel, 0, red, green, blue. Used for thumbnail images

        if endian == "little":
            endian = "<"
        else:
            endian = ">"

        dt = numpy.dtype(endian + dtyped[str(type_string)])

        return [x, y, z, header_size, dt]

    def get_info(self):
        if self.tags_dict == None:
            self.parse_file()
        image_list = self.tags_dict["ImageList"]["TagGroup1"]
        pixel_size = float(
            10
            * float(
                image_list["ImageData"]["Calibrations"]["Dimension"]["TagGroup0"][
                    "Scale"
                ]
            )
        )
        if "Microscope Info" in list(image_list["ImageTags"].keys()):
            voltage = (
                float(image_list["ImageTags"]["Microscope Info"]["Voltage"]) / 1000
            )
            mag = float(
                image_list["ImageTags"]["Microscope Info"]["Indicated Magnification"]
            )
        else:
            # print 'WARNING - No info in DM header.'
            voltage = 300.0
            mag = 105000.0
            pixel_size = 0.637542307377

        return [pixel_size, voltage, mag]

    def get_tilt_angles(self):
        if self.tags_dict == None:
            self.parse_file()
        try:
            data = self.tags_dict["ImageList"]["TagGroup1"]["ImageTags"]["Meta Data"][
                "Dimension info"
            ]["2"]["Data"]
            z = self.get_image_info()[2]

            tilt_angles = [None] * z
            for index, angle in data.items():
                tilt_angles[int(index)] = angle

            if len(data) < z:
                print(
                    "WARNING - number of tilt angles is less than expected. File is probably truncated."
                )
                if len(data) > 1:
                    myindex = numpy.nan
                    for index, angle in data.items():
                        myindex = int(index)
                        if (
                            tilt_angles[myindex + 1] is not None
                            and tilt_angles[myindex] is not None
                        ):
                            interval = float(tilt_angles[myindex + 1]) - float(
                                tilt_angles[myindex]
                            )
                            break
                    if myindex != numpy.nan:
                        minimum = tilt_angles[myindex] - myindex * interval
                    else:
                        interval = 2
                        minimum = -60
                else:
                    interval = 2
                    minimum = -60
                for i in range(z):
                    if tilt_angles[i] is None:
                        tilt_angles[i] = minimum + i * interval
                        print(
                            "Extrapolating tilt angle {0} to {1}".format(
                                i, tilt_angles[i]
                            )
                        )

        except KeyError:
            try:
                tilt_angles = self.tags_dict["ImageList"]["TagGroup1"]["ImageTags"][
                    "Microscope Info"
                ]["Stage Position"]["Stage Alpha"]
            except:
                z = self.get_image_info()[2]
                tilt_angles = [None] * z
                interval = 2
                minimum = -60
                for i in range(z):
                    tilt_angles[i] = minimum + i * interval

        return tilt_angles

    def get_tilt_axis_rotation(self):
        if self.tags_dict == None:
            self.parse_file()
        try:

            flipv = self.tags_dict["ImageList"]["TagGroup1"]["ImageTags"][
                "Acquisition"
            ]["Device"]["Configuration"]["Transpose"]["Vertical Flip"]
            fliph = self.tags_dict["ImageList"]["TagGroup1"]["ImageTags"][
                "Acquisition"
            ]["Device"]["Configuration"]["Transpose"]["Horizontal Flip"]
            flipd = self.tags_dict["ImageList"]["TagGroup1"]["ImageTags"][
                "Acquisition"
            ]["Device"]["Configuration"]["Transpose"]["Diagonal Flip"]

        except KeyError:

            flipv = fliph = flipd = 1

        if flipv == 1 and fliph == 0 and flipd == 1:
            return -90.0
        elif flipv == 0 and fliph == 0 and flipd == 0:
            return 180.0
        else:
            print(
                "WARNING - Cannot determine tilt-axis rotation from xyd-flip combination:",
                flipv,
                fliph,
                flipd,
            )
            return 0.0

    def pretty_print():
        if self.tags_dict == None:
            self.parse_file()
        pretty(self.tags_dict, 0)

    def write_info(self, metafile):  # Extract file base name and create meta file name
        if self.tags_dict == None:
            self.parse_file()

        # Open file for meta data
        dumpmeta = open(metafile, "w")

        # Write tags and values to file after removing extraneous characters
        numtags = len(self.tags_dict)
        for i in range(0, numtags):
            entry = str(list(self.tags_dict.items())[i])
            entry = entry.replace("(u'", "'")
            entry = entry.replace("u'", "'")
            entry = entry.replace("')", "'")
            dumpmeta.write(entry + "\n")

        # Close file for meta data
        dumpmeta.close

    def parse_header(self):
        self.dm_version = read_long(self.f, "big")
        if self.dm_version not in (3, 4):
            print(("File address:", dm_version[1]))
            raise NotImplementedError(
                "Currently we only support reading DM versions 3 and 4 but "
                "this file "
                "seems to be version %s " % self.dm_version
            )
        # self.skipif4()
        filesizeB = read_longlong(self.f, "big")
        is_little_endian = read_long(self.f, "big")

        if self.verbose is True:
            # filesizeMB = filesizeB[3] / 2.**20
            print(("DM version: %i" % self.dm_version))
            print(("size %i B" % filesizeB))
            print(("Is file Little endian? %s" % bool(is_little_endian)))
        if bool(is_little_endian):
            self.endian = "little"
        else:
            self.endian = "big"

    def parse_tags(self, ntags, group_name="root", group_dict={}):
        """Parse the DM file into a dictionary.

        """
        unnammed_data_tags = 0
        unnammed_group_tags = 0
        for tag in range(ntags):
            if self.verbose is True:
                print(("Reading tag name at address:", self.f.tell()))
            tag_header = self.parse_tag_header()
            tag_name = tag_header["tag_name"]

            skip = True if (group_name == "ImageData" and tag_name == "Data") else False
            if self.verbose is True:
                print(("Tag name:", tag_name[:20]))
                print(("Tag ID:", tag_header["tag_id"]))

            #            if tag_name != "ImageList":
            #                self.f.seek( int(tag_header['tag_size']), 1 )
            #                continue

            if tag_header["tag_id"] == 21:  # it's a TagType (DATA)
                if not tag_name:
                    tag_name = "Data%i" % unnammed_data_tags
                    unnammed_data_tags += 1

                if self.verbose is True:
                    print(("Reading data tag at address:", self.f.tell()))

                # Start reading the data
                self.check_data_tag_delimiter()  # Raises IOError if it is wrong
                # self.skipif4()
                infoarray_size = read_longlong(self.f, "big")
                if self.verbose:
                    print(("Infoarray size ", infoarray_size))
                # self.skipif4()
                if infoarray_size == 1:  # Simple type
                    if self.verbose:
                        print("Reading simple data")
                    etype = read_longlong(self.f, "big")
                    data = self.read_simple_data(etype)
                elif infoarray_size == 2:  # String
                    if self.verbose:
                        print("Reading string")
                    enctype = read_longlong(self.f, "big")
                    if enctype != 18:
                        raise IOError("Expected 18 (string), got %i" % enctype)
                    string_length = self.parse_string_definition()
                    data = self.read_string(string_length, skip=skip)
                elif infoarray_size == 3:  # Array of simple type
                    if self.verbose:
                        print("Reading simple array")
                    # Read array header
                    enctype = read_longlong(self.f, "big")
                    if enctype != 20:  # Should be 20 if it is an array
                        raise IOError("Expected 20 (string), got %i" % enctype)
                    size, enc_eltype = self.parse_array_definition()
                    data = self.read_array(size, enc_eltype, skip=skip)
                elif infoarray_size > 3:
                    enctype = read_longlong(self.f, "big")
                    if enctype == 15:  # It is a struct
                        if self.verbose:
                            print("Reading struct")
                        definition = self.parse_struct_definition()
                        if self.verbose:
                            print(("Struct definition ", definition))
                        data = self.read_struct(definition, skip=skip)
                    elif enctype == 20:  # It is an array of complex type
                        # Read complex array info
                        # The structure is
                        # 20 <4>, ?  <4>, enc_dtype <4>, definition <?>,
                        # size <4>
                        # self.skipif4()
                        enc_eltype = read_longlong(self.f, "big")
                        if enc_eltype == 15:  # Array of structs
                            if self.verbose:
                                print("Reading array of structs")
                            definition = self.parse_struct_definition()
                            # self.skipif4() # Padding?
                            size = read_longlong(self.f, "big")
                            if self.verbose:
                                print(("Struct definition: ", definition))
                                print(("Array size: ", size))
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"definition": definition},
                                skip=skip,
                            )
                        elif enc_eltype == 18:  # Array of strings
                            if self.verbose:
                                print("Reading array of strings")
                            string_length = self.parse_string_definition()
                            size = read_longlong(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"length": string_length},
                                skip=skip,
                            )
                        elif enc_eltype == 20:  # Array of arrays
                            if self.verbose:
                                print("Reading array of arrays")
                            el_length, enc_eltype = self.parse_array_definition()
                            size = read_longlong(self.f, "big")
                            data = self.read_array(
                                size=size,
                                enc_eltype=enc_eltype,
                                extra={"size": el_length},
                                skip=skip,
                            )

                else:  # Infoarray_size < 1
                    raise IOError("Invalided infoarray size ", infoarray_size)

                if self.verbose:
                    print(("Data: %s" % str(data)[:70]))
                group_dict[tag_name] = data

            elif tag_header["tag_id"] == 20:  # it's a TagGroup (GROUP)
                if not tag_name:
                    tag_name = "TagGroup%i" % unnammed_group_tags
                    unnammed_group_tags += 1
                if self.verbose is True:
                    print(("Reading Tag group at address:", self.f.tell()))
                ntags = self.parse_tag_group(skip4=3)[2]
                group_dict[tag_name] = {}
                self.parse_tags(
                    ntags=ntags, group_name=tag_name, group_dict=group_dict[tag_name]
                )
            else:
                print(("File address:", self.f.tell()))
                # raise DM3TagIDError(tag_header['tag_id'])
                raise Exception("ERROR")

    def get_data_reader(self, enc_dtype):
        # _data_type dictionary.
        # The first element of the InfoArray in the TagType
        # will always be one of _data_type keys.
        # the tuple reads: ('read bytes function', 'number of bytes', 'type')

        dtype_dict = {
            2: (read_short, 2, "h"),
            3: (read_long, 4, "l"),
            4: (read_ushort, 2, "H"),  # dm3 uses ushorts for unicode chars
            5: (read_ulong, 4, "L"),
            6: (read_float, 4, "f"),
            7: (read_double, 8, "d"),
            8: (read_boolean, 1, "B"),
            9: (read_char, 1, "b"),  # dm3 uses chars for 1-Byte signed integers
            10: (read_byte, 1, "b"),  # 0x0a
            11: (read_longlong, 8, "l"),  # Unknown, new in DM4
            12: (read_double, 8, "l"),  # Unknown, new in DM4
            15: (self.read_struct, None, "struct",),  # 0x0f
            18: (self.read_string, None, "c"),  # 0x12
            20: (self.read_array, None, "array"),  # 0x14
        }
        return dtype_dict[enc_dtype]

    def skipif4(self, n=1):
        if self.dm_version == 3:
            self.f.seek(4 * n, 1)

    def parse_array_definition(self):
        """Reads and returns the element type and length of the array.

        The position in the file must be just after the
        array encoded dtype.

        """
        # self.skipif4()
        enc_eltype = read_longlong(self.f, "big")
        # self.skipif4()
        length = read_longlong(self.f, "big")
        return length, enc_eltype

    def parse_string_definition(self):
        """Reads and returns the length of the string.

        The position in the file must be just after the
        string encoded dtype.
        """
        # self.skipif4()
        return read_longlong(self.f, "big")

    def parse_struct_definition(self):
        """Reads and returns the struct definition tuple.

        The position in the file must be just after the
        struct encoded dtype.

        """
        self.f.seek(8, 1)  # Skip the name length
        # self.skipif4(2)
        nfields = read_longlong(self.f, "big")
        definition = ()
        for ifield in range(nfields):
            self.f.seek(8, 1)
            # self.skipif4(2)
            definition += (read_longlong(self.f, "big"),)

        return definition

    def read_simple_data(self, etype):
        """Parse the data of the given DM3 file f
        with the given endianness (byte order).
        The infoArray iarray specifies how to read the data.
        Returns the tuple (file address, data).
        The tag data is stored in the platform's byte order:
        'little' endian for Intel, PC; 'big' endian for Mac, Motorola.
        If skip != 0 the data is actually skipped.
        """
        data = self.get_data_reader(etype)[0](self.f, self.endian)
        if isinstance(data, str):
            data = hyperspy.misc.utils.ensure_unicode(data)
        return data

    def read_string(self, length, skip=False):
        """Read a string defined by the infoArray iarray from
         file f with a given endianness (byte order).
        endian can be either 'big' or 'little'.

        If it's a tag name, each char is 1-Byte;
        if it's a tag data, each char is 2-Bytes Unicode,
        """
        if skip is True:
            offset = self.f.tell()
            self.f.seek(lenght, 1)
            return {
                "size": lenght,
                "size_bytes": size_bytes,
                "offset": offset,
                "endian": self.endian,
            }
        data = b""
        if self.endian == "little":
            s = L_char
        elif self.endian == "big":
            s = B_char
        for char in range(length):
            data += s.unpack(self.f.read(1))[0]
        try:
            data = data.decode("utf8")
        except:
            # Sometimes the dm3 file strings are encoded in latin-1
            # instead of utf8
            data = data.decode("latin-1", errors="ignore")
        return data

    def read_struct(self, definition, skip=False):
        """Read a struct, defined by iarray, from file f
        with a given endianness (byte order).
        Returns a list of 2-tuples in the form
        (fieldAddress, fieldValue).
        endian can be either 'big' or 'little'.
        
        """
        field_value = []
        size_bytes = 0
        offset = self.f.tell()
        for dtype in definition:
            if dtype in self.simple_type:
                if skip is False:
                    data = self.get_data_reader(dtype)[0](self.f, self.endian)
                    field_value.append(data)
                else:
                    sbytes = self.get_data_reader(dtype)[1]
                    self.f.seek(sbytes, 1)
                    size_bytes += sbytes
            else:
                raise DM3DataTypeError(dtype)
        if skip is False:
            return tuple(field_value)
        else:
            return {
                "size": len(definition),
                "size_bytes": size_bytes,
                "offset": offset,
                "endian": self.endian,
            }

    def read_array(self, size, enc_eltype, extra=None, skip=False):
        """Read an array, defined by iarray, from file f
        with a given endianness (byte order).
        endian can be either 'big' or 'little'.

        """
        eltype = self.get_data_reader(enc_eltype)[0]  # same for all elements
        if skip is True:
            if enc_eltype not in self._complex_type:
                size_bytes = self.get_data_reader(enc_eltype)[1] * size
                data = {
                    "size": size,
                    "endian": self.endian,
                    "size_bytes": size_bytes,
                    "offset": self.f.tell(),
                }
                self.f.seek(size_bytes, 1)  # Skipping data
            else:
                data = eltype(skip=skip, **extra)
                self.f.seek(data["size_bytes"] * (size - 1), 1)
                data["size"] = size
                data["size_bytes"] *= size
        else:
            if enc_eltype in self.simple_type:  # simple type
                data = [eltype(self.f, self.endian) for element in range(size)]
                if enc_eltype == 4 and data:  # it's actually a string
                    data = "".join([chr(i) for i in data])
            elif enc_eltype in self._complex_type:
                data = [eltype(**extra) for element in range(size)]
        return data

    def parse_tag_group(self, skip4=1):
        """Parse the root TagGroup of the given DM3 file f.
        Returns the tuple (is_sorted, is_open, n_tags).
        endian can be either 'big' or 'little'.
        """
        is_sorted = read_byte(self.f, "big")
        is_open = read_byte(self.f, "big")
        # self.skipif4(n=skip4)
        n_tags = read_longlong(self.f, "big")
        return bool(is_sorted), bool(is_open), n_tags

    def find_next_tag(self):
        while read_byte(self.f, "big") not in (20, 21):
            continue
        location = self.f.tell() - 1
        self.f.seek(location)
        tag_id = read_byte(self.f, "big")
        self.f.seek(location)
        tag_header = self.parse_tag_header()
        if tag_id == 20:
            print(("Tag header length", tag_header["tag_name_length"]))
            if not 20 > tag_header["tag_name_length"] > 0:
                print("Skipping id 20")
                self.f.seek(location + 1)
                self.find_next_tag()
            else:
                self.f.seek(location)
                return
        else:
            try:
                self.check_data_tag_delimiter()
                self.f.seek(location)
                return
            except DM3TagTypeError:
                self.f.seek(location + 1)
                print("Skipping id 21")
                self.find_next_tag()

    def find_next_data_tag(self):
        while read_byte(self.f, "big") != 21:
            continue
        position = self.f.tell() - 1
        self.f.seek(position)
        tag_header = self.parse_tag_header()
        try:
            self.check_data_tag_delimiter()
            self.f.seek(position)
        except DM3TagTypeError:
            self.f.seek(position + 1)
            self.find_next_data_tag()

    def parse_tag_header(self):
        tag_id = read_byte(self.f, "big")
        tag_name_length = read_short(self.f, "big")
        tag_name = self.read_string(tag_name_length)
        tag_size = read_longlong(self.f, "big")
        return {
            "tag_id": tag_id,
            "tag_name_length": tag_name_length,
            "tag_name": tag_name,
            "tag_size": tag_size,
        }

    def check_data_tag_delimiter(self):
        # self.skipif4(2)
        delimiter = self.read_string(4)
        if delimiter != "%%%%":
            raise DM3TagTypeError(delimiter)

    def get_image_dictionaries(self):
        """Returns the image dictionaries of all images in the file except
        the thumbnails.

        Returns
        -------
        dict, None

        """
        if "ImageList" not in self.tags_dict:
            return None
        if "Thumbnails" in self.tags_dict:
            thumbnail_idx = [
                t["ImageIndex"] for key, t in self.tags_dict["Thumbnails"].items()
            ]
        else:
            thumbnail_idx = []
        images = [
            image
            for key, image in self.tags_dict["ImageList"].items()
            if not int(key.replace("TagGroup", "")) in thumbnail_idx
        ]
        return images


class ImageObject(object):
    def __init__(self, imdict, file, order="C", record_by=None):
        self.imdict = DictionaryBrowser(imdict)
        self.file = file
        self._order = order if order else "C"
        self._record_by = record_by

    @property
    def shape(self):
        dimensions = self.imdict.ImageData.Dimensions
        shape = tuple([dimension[1] for dimension in dimensions])
        return shape[::-1]  # DM uses image indexing X, Y, Z...

    @property
    def offsets(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        origins = np.array([dimension[1].Origin for dimension in dimensions])
        return -1 * origins[::-1] * self.scales

    @property
    def scales(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        return np.array([dimension[1].Scale for dimension in dimensions])[::-1]

    @property
    def units(self):
        dimensions = self.imdict.ImageData.Calibrations.Dimension
        return tuple(
            [
                dimension[1].Units if dimension[1].Units else ""
                for dimension in dimensions
            ]
        )[::-1]

    @property
    def names(self):
        names = [t.Undefined] * len(self.shape)
        indices = list(range(len(self.shape)))
        if self.signal_type == "EELS":
            if "eV" in self.units:
                names[indices.pop(self.units.index("eV"))] = "Energy loss"
        elif self.signal_type in ("EDS", "EDX"):
            if "keV" in self.units:
                names[indices.pop(self.units.index("keV"))] = "Energy"
        for index, name in zip(indices[::-1], ("x", "y", "z")):
            names[index] = name
        return names

    @property
    def title(self):
        if "Name" in self.imdict:
            return self.imdict.Name
        else:
            return ""

    @property
    def record_by(self):
        if self._record_by is not None:
            return self._record_by
        if len(self.scales) == 1:
            return "spectrum"
        elif (
            (
                "ImageTags.Meta_Data.Format" in self.imdict
                and self.imdict.ImageTags.Meta_Data.Format == "Spectrum image"
            )
            or ("ImageTags.spim" in self.imdict)
        ) and len(self.scales) == 2:
            return "spectrum"
        else:
            return "image"

    @property
    def to_spectrum(self):
        if (
            (
                "ImageTags.Meta_Data.Format" in self.imdict
                and self.imdict.ImageTags.Meta_Data.Format == "Spectrum image"
            )
            or ("ImageTags.spim" in self.imdict)
        ) and len(self.scales) > 2:
            return True
        else:
            return False

    @property
    def order(self):
        return self._order

    @property
    def intensity_calibration(self):
        ic = self.imdict.ImageData.Calibrations.Brightness.as_dictionary()
        if not ic["Units"]:
            ic["Units"] = ""
        return ic

    @property
    def dtype(self):
        # Image data types (Image Object chapter on DM help)#
        # key = DM data type code
        # value = numpy data type
        if self.imdict.ImageData.DataType == 4:
            raise NotImplementedError("Reading data of this type is not implemented.")

        imdtype_dict = {
            0: "not_implemented",  # null
            1: "int16",
            2: "float32",
            3: "complex64",
            5: "float32",  # not numpy: 8-Byte packed complex (FFT data)
            6: "uint8",
            7: "int32",
            8: np.dtype(
                {"names": ["B", "G", "R", "A"], "formats": ["u1", "u1", "u1", "u1"]}
            ),
            9: "int8",
            10: "uint16",
            11: "uint32",
            12: "float64",
            13: "complex128",
            14: "bool",
            23: np.dtype(
                {"names": ["B", "G", "R", "A"], "formats": ["u1", "u1", "u1", "u1"]}
            ),
            27: "complex64",  # not numpy: 8-Byte packed complex (FFT data)
            28: "complex128",  # not numpy: 16-Byte packed complex (FFT data)
        }
        return imdtype_dict[self.imdict.ImageData.DataType]

    @property
    def signal_type(self):
        if "ImageTags.Meta_Data.Signal" in self.imdict:
            return self.imdict.ImageTags.Meta_Data.Signal
        elif "ImageTags.spim.eels" in self.imdict:  # Orsay's tag group
            return "EELS"
        else:
            return ""

    def _get_data_array(self):
        self.file.seek(self.imdict.ImageData.Data.offset)
        count = self.imdict.ImageData.Data.size
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            count = int(count / 2)
        return np.fromfile(self.file, dtype=self.dtype, count=count)

    @property
    def size(self):
        if self.imdict.ImageData.DataType in (27, 28):  # Packed complex
            if self.imdict.ImageData.Data.size % 2:
                raise IOError(
                    "ImageData.Data.size should be an even integer for "
                    "this datatype."
                )
            else:
                return int(self.imdict.ImageData.Data.size / 2)
        else:
            return self.imdict.ImageData.Data.size

    def get_data(self):
        if isinstance(self.imdict.ImageData.Data, np.ndarray):
            return self.imdict.ImageData.Data
        data = self._get_data_array()
        if self.imdict.ImageData.DataType in (27, 28):  # New packed complex
            return self.unpack_new_packed_complex(data)
        elif self.imdict.ImageData.DataType == 5:  # Old packed compled
            return self.unpack_packed_complex(data)
        elif self.imdict.ImageData.DataType in (8, 23):  # ABGR
            # Reorder the fields
            data = (
                np.hstack(
                    (
                        data[["B", "G", "R"]].view(("u1", 3))[..., ::-1],
                        data["A"].reshape(-1, 1),
                    )
                )
                .view({"names": ("R", "G", "B", "A"), "formats": ("u1",) * 4})
                .copy()
            )
        return data.reshape(self.shape, order=self.order)

    def unpack_new_packed_complex(self, data):
        packed_shape = (self.shape[0], int(self.shape[1] / 2 + 1))
        data = data.reshape(packed_shape, order=self.order)
        return np.hstack((data[:, ::-1], np.conjugate(data[:, 1:-1])))

    def unpack_packed_complex(self, tmpdata):
        shape = self.shape
        if shape[0] != shape[1] or len(shape) > 2:
            msg = "Packed complex format works only for a 2Nx2N image"
            msg += " -> width == height"
            print(msg)
            raise IOError(
                "Unable to read this DM file in packed complex format. "
                "Pleare report the issue to the Hyperspy developers providing"
                " the file if possible"
            )
        N = int(self.shape[0] / 2)  # think about a 2Nx2N matrix
        # create an empty 2Nx2N ndarray of complex
        data = np.zeros(shape, dtype="complex64")

        # fill in the real values:
        data[N, 0] = tmpdata[0]
        data[0, 0] = tmpdata[1]
        data[N, N] = tmpdata[2 * N ** 2]  # Nyquist frequency
        data[0, N] = tmpdata[2 * N ** 2 + 1]  # Nyquist frequency

        # fill in the non-redundant complex values:
        # top right quarter, except 1st column
        for i in range(N):  # this could be optimized
            start = 2 * i * N + 2
            stop = start + 2 * (N - 1) - 1
            step = 2
            realpart = tmpdata[start:stop:step]
            imagpart = tmpdata[start + 1 : stop + 1 : step]
            data[i, N + 1 : 2 * N] = realpart + imagpart * 1j
        # 1st column, bottom left quarter
        start = 2 * N
        stop = start + 2 * N * (N - 1) - 1
        step = 2 * N
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1 : stop + 1 : step]
        data[N + 1 : 2 * N, 0] = realpart + imagpart * 1j
        # 1st row, bottom right quarter
        start = 2 * N ** 2 + 2
        stop = start + 2 * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1 : stop + 1 : step]
        data[N, N + 1 : 2 * N] = realpart + imagpart * 1j
        # bottom right quarter, except 1st row
        start = stop + 1
        stop = start + 2 * N * (N - 1) - 1
        step = 2
        realpart = tmpdata[start:stop:step]
        imagpart = tmpdata[start + 1 : stop + 1 : step]
        complexdata = realpart + imagpart * 1j
        data[N + 1 : 2 * N, N : 2 * N] = complexdata.reshape(N - 1, N, order=self.order)

        # fill in the empty pixels: A(i)(j) = A(2N-i)(2N-j)*
        # 1st row, top left quarter, except 1st element
        data[0, 1:N] = np.conjugate(data[0, -1:-N:-1])
        # 1st row, bottom left quarter, except 1st element
        data[N, 1:N] = np.conjugate(data[N, -1:-N:-1])
        # 1st column, top left quarter, except 1st element
        data[1:N, 0] = np.conjugate(data[-1:-N:-1, 0])
        # 1st column, top right quarter, except 1st element
        data[1:N, N] = np.conjugate(data[-1:-N:-1, N])
        # top left quarter, except 1st row and 1st column
        data[1:N, 1:N] = np.conjugate(data[-1:-N:-1, -1:-N:-1])
        # bottom left quarter, except 1st row and 1st column
        data[N + 1 : 2 * N, 1:N] = np.conjugate(data[-N - 1 : -2 * N : -1, -1:-N:-1])

        return data

    def get_axes_dict(self):
        return [
            {
                "name": name,
                "size": size,
                "index_in_array": i,
                "scale": scale,
                "offset": offset,
                "units": str(units),
            }
            for i, (name, size, scale, offset, units) in enumerate(
                zip(self.names, self.shape, self.scales, self.offsets, self.units)
            )
        ]

    def get_mapped_parameters(self, mapped_parameters={}):
        mapped_parameters["title"] = self.title
        mapped_parameters["record_by"] = self.record_by
        mapped_parameters["signal_type"] = self.signal_type
        return mapped_parameters


mapping = {
    "ImageList.TagGroup0.ImageTags.EELS.Experimental_Conditions.Collection_semi_angle_mrad": (
        "TEM.EELS.collection_angle",
        None,
    ),
    "ImageList.TagGroup0.ImageTags.EELS.Experimental_Conditions.Convergence_semi_angle_mrad": (
        "TEM.convergence_angle",
        None,
    ),
    "ImageList.TagGroup0.ImageTags.Acquisition.Parameters.Detector.exposure_s": (
        "TEM.dwell_time",
        None,
    ),
    "ImageList.TagGroup0.ImageTags.Microscope_Info.Voltage": (
        "TEM.beam_energy",
        lambda x: x / 1e3,
    ),
}


def file_reader(filename, record_by=None, order=None, verbose=False):
    """Reads a DM3 file and loads the data into the appropriate class.
    data_id can be specified to load a given image within a DM3 file that
    contains more than one dataset.
    
    Parameters
    ----------
    record_by: Str
        One of: SI, Image
    order: Str
        One of 'C' or 'F'

    """

    with open(filename, "rb") as f:
        dm = DigitalMicrographReader(f, verbose=verbose)
        dm.parse_file()
        images = [
            ImageObject(imdict, f, order=order, record_by=record_by)
            for imdict in dm.get_image_dictionaries()
        ]
        imd = []
        del dm.tags_dict["ImageList"]
        dm.tags_dict["ImageList"] = {}

        for image in images:
            dm.tags_dict["ImageList"]["TagGroup0"] = image.imdict.as_dictionary()
            axes = image.get_axes_dict()
            mp = image.get_mapped_parameters()
            mp["original_filename"] = os.path.split(filename)[1]
            post_process = []
            if image.to_spectrum is True:
                post_process.append(lambda s: s.to_spectrum())
            post_process.append(lambda s: s.squeeze())
            imd.append(
                {
                    "data": image.get_data(),
                    "axes": axes,
                    "mapped_parameters": mp,
                    "original_parameters": dm.tags_dict,
                    "post_process": post_process,
                    "mapping": mapping,
                }
            )

    return imd


def pretty(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key.encode("utf-8")))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process tilt-series")
    parser.add_argument("-file", help="Input file name", type=str, required=True)
    parser.add_argument(
        "-header", help="Show header", action="store_true", default=False
    )
    parser.add_argument(
        "-tilts", help="Show tilt angle information", action="store_true", default=False
    )
    args = parser.parse_args()

    f = open(args.file, "rb")
    dm = DigitalMicrographReader(f, verbose=False)

    if args.header:
        dm.parse_file()
        pretty(dm.tags_dict, 0)

    print("Pixel size, Voltage, Magnification = ", dm.get_info())
    print("X, Y, Z, offset = ", dm.get_image_info())
    if args.tilts:
        print("Tilt angle(s) = ", dm.get_tilt_angles())

    f.close()

    ## Extract file base name and create meta file name
    # metafile = "meta.txt"
    #
    ## Open file for meta data
    # dumpmeta = open(metafile, 'w')
    #
    ## Write tags and values to file after removing extraneous characters
    # numtags = len(dm.tags_dict)
    # for i in range(0, numtags):
    # 	entry = str(dm.tags_dict.items()[i])
    # 	entry = entry.replace("(u'", "'")
    # 	entry = entry.replace("u'", "'")
    # 	entry = entry.replace("')", "'")
    # 	dumpmeta.write(entry + "\n")
    #
    ## Close file for meta data
    # dumpmeta.close
