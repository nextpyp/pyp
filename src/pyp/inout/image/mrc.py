#!/usr/bin/env python

# Downloaded from http://ami.scripps.edu/software/tiltpicker/
# tiltpicker/pyami/mrc.py
# This file had no copyright notice, but another in the same directory had this:
#
#
# COPYRIGHT:
# The Leginon software is Copyright 2003
# The Scripps Research Institute, La Jolla, CA
# For terms of the license agreement
# see	http://ami.scripps.edu/software/leginon-license


"""MRC I/O functions

write(a, filename, header=None)
Write your numpy ndarray object to an MRC file.
        a - the numpy ndarray object
    filename - the filename you want to write to
    header - (optional) dictionary of additional header information

read(filename)
    Read MRC file into a numpy ndarray object.
    filename - the MRC filename

mmap(filename)  
Open MRC as a memory mapped file.  This is the most efficient way
to read data if you only need to access part of a large MRC file.
Only the parts you actually access are read from the disk into memory.
        filename - the MRC filename

numarray_read(filename)
convenience function if you want your array returned as numarray instead.
numarray_write(a, filename, header=None)
convenience function if you want to write a numarray array instead.
"""

from re import L
import shutil
import os
import sys
from functools import reduce

import numpy

from pyp.inout.image.utils import arraystats, weakattr

from pyp.system.utils import get_frealign_paths
from pyp.system.local_run import run_shell_command

#### for numarray compatibility
try:
    import numarray
except:
    numarray = None
if numarray is not None:

    def numarray_read(filename):
        a1 = read(filename)
        a2 = numarray.array(a1)
        return a1

    def numarray_write(a, filename, header=None):
        a2 = numpy.asarray(a)
        write(a2, filename, header)


## mapping of MRC mode to numpy type
mrc2numpy = {
    0: numpy.uint8,
    1: numpy.int16,
    2: numpy.float32,
    # 	3:  complex made of two int16.  No such thing in numpy
    #     however, we could manually build a complex array by reading two
    #     int16 arrays somehow.
    4: numpy.complex64,
    6: numpy.uint16,  # according to UCSF
    101: numpy.uint8,  # s4-bit format support
}

## mapping of numpy type to MRC mode
numpy2mrc = {
    ## convert these to int8
    numpy.uint8: 0,
    bool: 0,
    numpy.bool_: 0,
    ## convert these to int16
    numpy.int16: 1,
    numpy.int8: 1,
    ## convert these to float32
    numpy.float32: 2,
    numpy.float64: 2,
    numpy.int32: 2,
    numpy.int64: 2,
    int: 2,
    numpy.uint32: 2,
    numpy.uint64: 2,
    ## convert these to complex64
    complex: 4,
    numpy.complex64: 4,
    numpy.complex128: 4,
    ## convert these to uint16
    numpy.uint16: 6,
}

## structure of the image 2000 MRC header
## This is a sequence of fields where each field is defined by a sequence:
##  (name, type, default, length)
##    length is only necessary for strings
##    type can be one of: 'int32', 'float32', 'string'
header_fields = (
    ("nx", "int32"),
    ("ny", "int32"),
    ("nz", "int32"),
    ("mode", "int32"),
    ("nxstart", "int32"),
    ("nystart", "int32"),
    ("nzstart", "int32"),
    ("mx", "int32"),
    ("my", "int32"),
    ("mz", "int32"),
    ("xlen", "float32"),
    ("ylen", "float32"),
    ("zlen", "float32"),
    ("alpha", "float32"),
    ("beta", "float32"),
    ("gamma", "float32"),
    ("mapc", "int32"),
    ("mapr", "int32"),
    ("maps", "int32"),
    ("amin", "float32"),
    ("amax", "float32"),
    ("amean", "float32"),
    ("ispg", "int32"),
    ("nsymbt", "int32"),
    ("extra", "string", 100),
    ("xorigin", "float32"),
    ("yorigin", "float32"),
    ("zorigin", "float32"),
    ("map", "string", 4),
    ("byteorder", "int32"),
    ("rms", "float32"),
    ("nlabels", "int32"),
    ("label0", "string", 80),
    ("label1", "string", 80),
    ("label2", "string", 80),
    ("label3", "string", 80),
    ("label4", "string", 80),
    ("label5", "string", 80),
    ("label6", "string", 80),
    ("label7", "string", 80),
    ("label8", "string", 80),
    ("label9", "string", 80),
)

## Boulder format of stack mrc header
for i, x in enumerate(header_fields):
    if x[0] == "extra":
        break
header_fields_stack = list(header_fields[:i])
header_fields_stack.extend(
    [
        ("dvid", "uint16"),
        ("nblank", "uint16"),
        ("itst", "int32"),
        ("blank", "string", 24),
        ("nintegers", "uint16"),
        ("nfloats", "uint16"),
        ("sub", "uint16"),
        ("zfac", "uint16"),
        ("min2", "float32"),
        ("max2", "float32"),
        ("min3", "float32"),
        ("max3", "float32"),
        ("min4", "float32"),
        ("max4", "float32"),
        ("type", "uint16"),
        ("lensum", "uint16"),
        ("nd1", "uint16"),
        ("nd2", "uint16"),
        ("vd1", "uint16"),
        ("vd2", "uint16"),
        ("min5", "float32"),
        ("max5", "float32"),
        ("numtimes", "uint16"),
        ("imgseq", "uint16"),
        ("xtilt", "float32"),
        ("ytilt", "float32"),
        ("ztilt", "float32"),
        ("numwaves", "uint16"),
        ("wave1", "uint16"),
        ("wave2", "uint16"),
        ("wave3", "uint16"),
        ("wave4", "uint16"),
        ("wave5", "uint16"),
        ("xorigin", "float32"),
        ("yorigin", "float32"),
        ("zorigin", "float32"),
        ("nlabels", "int32"),
        ("label0", "string", 80),
        ("label1", "string", 80),
        ("label2", "string", 80),
        ("label3", "string", 80),
        ("label4", "string", 80),
        ("label5", "string", 80),
        ("label6", "string", 80),
        ("label7", "string", 80),
        ("label8", "string", 80),
        ("label9", "string", 80),
    ]
)

header_fields_extended = [
    ("stagealpha", "float32"),
    ("stagebeta", "float32"),
    ("stagex", "float32"),
    ("stagey", "float32"),
    ("stagez", "float32"),
    ("shiftx", "float32"),
    ("shifty", "float32"),
    ("defocus", "float32"),
    ("exposuretime", "float32"),
    ("meanintensity", "float32"),
    ("tiltaxis", "float32"),
    ("pixelsize", "float32"),
    ("magnification", "float32"),
    ("reserved", "string", 36),
]


def printHeader(headerdict):
    for field in header_fields:
        name = field[0]
        value = headerdict[name]
        print(("{-10}:  {}".format(name, value)))


def zeros(n):
    """
Create n bytes of data initialized to zeros, returned as a python string.
	"""
    a = numpy.zeros(n, dtype=int8dtype)
    return a.tobytes()


def newHeader(header_fields=header_fields):
    """
Return a new initialized header dictionary.
All fields are initialized to zeros.
	"""
    header = {}
    for field in header_fields:
        name = field[0]
        type = field[1]
        if type == "string":
            length = field[2]
            header[name] = zeros(length)
        else:
            header[name] = 0
    return header


intbyteorder = {0x11110000: "big", 0x44440000: "little"}
byteorderint = {"big": 0x11110000, "little": 0x44440000}


def isSwapped(headerbytes):
    """
Detect byte order (endianness) of MRC file based on one or more tests on
the header data.
	"""
    ### check for a valid machine stamp in header, with or without byteswap
    stampswapped = None
    machstamp = headerbytes[212:216]
    machstamp = numpy.fromstring(machstamp, dtype="int32", count=1)
    machstampint = machstamp[0]
    if machstampint in intbyteorder:
        stampswapped = False
    else:
        machstamp = machstamp.byteswap()
        machstampint = machstamp[0]
        if machstampint in intbyteorder:
            stampswapped = True

    ### check for valid mode, with or without byteswap
    mode = headerbytes[12:16]
    mode = numpy.fromstring(mode, dtype="int32", count=1)
    modeint = mode[0]
    modeswapped = None
    if modeint in mrc2numpy:
        modeswapped = False
    else:
        mode = mode.byteswap()
        modeint = mode[0]
        if modeint in mrc2numpy:
            modeswapped = True

    ### final verdict on whether it is swapped
    if stampswapped is None:
        swapped = modeswapped
    elif modeswapped is None:
        swapped = stampswapped
    elif modeswapped == stampswapped:
        swapped = modeswapped
    else:
        swapped = None
    return swapped


def parseHeader(headerbytes):
    """
Parse the 1024 byte MRC header into a header dictionary.
	"""
    ## header is comprised of int32, float32, and text labels.
    itype = numpy.dtype("int32")
    ftype = numpy.dtype("float32")

    ## check if data needs to be byte swapped
    swapped = isSwapped(headerbytes)
    if swapped:
        itype = itype.newbyteorder()
        ftype = ftype.newbyteorder()

    ## Convert 1k header into both floats and ints to make it easy
    ## to extract all the info.
    ## Only convert first 224 bytes into numbers because the
    ## remainder of data are text labels
    headerarray = {}
    headerarray["float32"] = numpy.fromstring(headerbytes, dtype=ftype, count=224)
    headerarray["int32"] = numpy.fromstring(headerbytes, dtype=itype, count=224)

    ## fill in header dictionary with all the info
    newheader = {}
    pos = 0

    for field in header_fields:
        name = field[0]
        type = field[1]
        if type == "string":
            length = field[2]
            newheader[name] = headerbytes[pos : pos + length]
        else:
            length = 4
            word = int(pos / 4)
            newheader[name] = headerarray[type][word]
        pos += length

    ## Save some numpy specific info (not directly related to MRC).
    ## numpy dtype added to header dict because it includes both the
    ## basic type (from MRC "mode") and also the byte order, which has
    ## been determined independent from the byte order designation in the
    ## header, which may be invalid.  This allows the data to be read
    ## properly.  Also figure out the numpy shape of the data from dimensions.
    dtype = numpy.dtype(mrc2numpy[newheader["mode"]])
    if swapped:
        dtype = dtype.newbyteorder()
    newheader["dtype"] = dtype
    if newheader["nz"] > 1:
        ## 3D data
        shape = (newheader["nz"], newheader["ny"], newheader["nx"])
    elif newheader["ny"] > 1:
        ## 2D data
        shape = (newheader["ny"], newheader["nx"])
    else:
        ## 1D data
        shape = (newheader["nx"],)
    newheader["shape"] = shape

    return newheader


def updateHeaderDefaults(header):
    header["alpha"] = 90
    header["beta"] = 90
    header["gamma"] = 90
    header["mapc"] = 1
    header["mapr"] = 2
    header["maps"] = 3
    header["map"] = "MAP "
    header["byteorder"] = byteorderint[sys.byteorder]


def updateHeaderUsingArray(header, a):
    """
	Fills in values of MRC header dictionary using the given array.
	"""
    ndims = len(a.shape)
    nx = a.shape[-1]
    ny = nz = 1
    if ndims > 1:
        ny = a.shape[-2]
        if ndims > 2:
            nz = a.shape[-3]
    header["nx"] = nx
    header["ny"] = ny
    header["nz"] = nz

    mode = numpy2mrc[a.dtype.type]
    header["mode"] = mode

    header["mx"] = nx
    header["my"] = ny
    header["mz"] = nz

    try:
        psize = weakattr.get(a, "pixelsize")
    except AttributeError:
        header["xlen"] = nx
        header["ylen"] = ny
        header["zlen"] = nz
    else:
        header["xlen"] = nx * psize["x"]
        header["ylen"] = ny * psize["y"]
        header["zlen"] = nz * psize["x"]

    stats = arraystats.all(a)
    header["amin"] = stats["min"]
    header["amax"] = stats["max"]
    header["amean"] = stats["mean"]
    header["rms"] = stats["std"]

    ### changed next lines to be equivalent to proc3d origin=0,0,0
    header["xorigin"] = 0
    header["yorigin"] = 0
    header["zorigin"] = 0
    if ndims < 4:
        header["nxstart"] = 0
        header["nystart"] = 0
        header["nzstart"] = 0
    else:
        header["nxstart"] = nx / -2
        header["nystart"] = ny / -2
        header["nzstart"] = nz / -2


int32dtype = numpy.dtype("int32")
uint16dtype = numpy.dtype("uint16")
float32dtype = numpy.dtype("float32")
int8dtype = numpy.dtype("int8")


def valueToFloat(value):
    """
return the string representation of a float value
	"""
    a = numpy.array(value, dtype=float32dtype)
    return a.tobytes()


def valueToInt(value):
    """
return the string representation of an int value
	"""
    a = numpy.array(value, dtype=int32dtype)
    return a.tobytes()


def valueTouint16(value):
    """
return the string representation of an int value
	"""
    a = numpy.array(value, dtype=uint16dtype)
    return a.tobytes()


def makeHeaderData(h, header_fields=header_fields):
    """
Create a 1024 byte header string from a header dictionary.
	"""
    fields = []
    for field in header_fields:
        name = field[0]
        type = field[1]
        if name in h:
            value = h[name]
        else:
            value = 0
        if type == "string":
            length = field[2]
            if isinstance(value, str):
                s = value.encode().strip()
            else:
                s = value
            nzeros = length - len(s)
            # import pdb; pdb.set_trace()
            fullfield = s + zeros(nzeros)
            fields.append(fullfield)
        elif type == "int32":
            fields.append(valueToInt(value))
        elif type == "float32":
            fields.append(valueToFloat(value))
        elif type == "uint16":
            fields.append(valueTouint16(value))

    headerbytes = b"".join(fields)
    return headerbytes


def asMRCtype(a):
    """
If necessary, convert a numpy ndarray to type that is compatible
with MRC.
	"""
    if not isinstance(a, numpy.ndarray):
        raise TypeError("Value must be a numpy array")

    t = a.dtype.type
    if t in numpy2mrc:
        numtype = t
    else:
        raise TypeError("Invalid Numeric array type for MRC conversion: %s" % (t,))
    dtype = numpy.dtype(mrc2numpy[numpy2mrc[numtype]])
    narray = numpy.asarray(a, dtype=dtype)
    return narray


def readDataFromFile(fobj, headerdict):
    """
	Read data portion of MRC file from the file object fobj.
	Both mrcmode and shape have been determined from the MRC header.
	fobj already points to beginning of data, not header.
	Returns a new numpy ndarray object.
	"""
    shape = headerdict["shape"]
    datalen = reduce(numpy.multiply, shape)
    if len(shape) == 2:
        datalen = int(shape[0]) * int(shape[1])
    else:
        datalen = int(shape[0]) * int(shape[1]) * int(shape[2])
    a = numpy.fromfile(fobj, dtype=headerdict["dtype"], count=datalen)
    a.shape = shape
    return a


def write(a, filename, header=None):
    """
Write ndarray to a file
a = numpy ndarray to be written
filename = filename of MRC
header (optional) = dictionary of header parameters
Always saves in the native byte order.
	"""

    h = newHeader()
    updateHeaderDefaults(h)
    updateHeaderUsingArray(h, a)

    if header is not None:
        h.update(header)

    headerbytes = makeHeaderData(h)
    f = open(filename, "wb")
    f.write(headerbytes)

    appendArray(a, f)

    f.close()


def mainStackHeader(oneheader, z):
    newheader = newHeader(header_fields=header_fields_stack)
    newheader.update(oneheader)
    newheader["nz"] = z
    newheader["mz"] = z
    newheader["zlen"] = z
    newheader["zorigin"] = z / 2.0
    newheader["nsymbt"] = z * 88
    newheader["nintegers"] = 0
    newheader["nfloats"] = 22
    return newheader


def extendedHeader(tilt):
    newheader = {}
    newheader["stagealpha"] = tilt
    ## other fields...

    return newheader


def stack(inputfiles, tilts, outputfile):
    # read first image to use as main header
    firstheader = readHeaderFromFile(inputfiles[0])
    newheader = mainStackHeader(firstheader, len(tilts))

    # write main header
    headerbytes = makeHeaderData(newheader, header_fields=header_fields_stack)
    f = open(outputfile, "wb")
    f.write(headerbytes)

    # write zeros for all extended headers
    extended_length = len(tilts) * 88
    f.write(zeros(extended_length))

    # write extended headers and data
    extheaderpos = 1024
    for inputfile, tilt in zip(inputfiles, tilts):
        data = read(inputfile)

        f.seek(extheaderpos)
        extheaderpos += 88
        newheader = extendedHeader(tilt)
        headerbytes = makeHeaderData(newheader, header_fields=header_fields_extended)
        f.write(headerbytes)
        appendArray(data, f)
    f.close()


# Merge multiple .mrc files into single stack
# without loading into memory
def merge(inputfiles, outputfile):
    # print(inputfiles)
    # write main header
    h = readHeaderFromFile(inputfiles[0])
    headerbytes = makeHeaderData(h)
    f = open(outputfile, "wb")
    f.write(headerbytes)

    # compute total number of images
    total = 0

    for inputfile in inputfiles:
        if os.path.isfile(inputfile):
            data = read(inputfile)
            appendArray(data, f)
            if data.ndim > 2:
                total += data.shape[0]
            else:
                total += 1

    # update header with total number of images
    h["nz"] = total
    h["mz"] = total
    headerbytes = makeHeaderData(h)

    f.seek(0, 0)
    f.write(headerbytes)
    f.close()


def merge_fast(inputfiles, outputfile, remove=False):

    """
            **   Welcome to AppendStacks   **

                Version : 1.00
                Compiled : Dec 15 2022
        Library Version : 2.0.0-alpha--1--dirty
            From Branch : (HEAD
                    Mode : Interactive

    Input image file name #1 [input_stack1.mrc]        : stack_1.mrc
    Input image file name #2 [input_stack2.mrc]        : stack_2.mrc

    Adding Images...

    100% [=================] done! (0h:00m00s)
    """

    if not remove:
        shutil.copy(inputfiles[0],outputfile)
        inputfiles[0] = outputfile

    append_stacks = f"{get_frealign_paths()['cistem2']}/append_stacks"
    for stack in inputfiles[1:]:
        command = f"""
{append_stacks} << EOF
{inputfiles[0]}
{stack}
EOF
        """
        [output, error] = run_shell_command(command, verbose=False)
        if remove:
            try:
                os.remove(stack)
            except:
                pass

    if remove:
        shutil.move(inputfiles[0],outputfile)

def repeatFile(outputfile, times):

    data = read(outputfile)

    # write main header
    h = readHeaderFromFile(outputfile)
    headerbytes = makeHeaderData(h)
    f = open(outputfile, "rb+")
    f.seek(0, 2)

    for i in range(times - 1):
        appendArray(data, f)

    # update header with total number of images
    h["nz"] *= times
    h["mz"] *= times
    headerbytes = makeHeaderData(h)

    f.seek(0, 0)
    f.write(headerbytes)
    f.close()


# Extract given slices from input stack into new one
def extract_slices(inputfile, indexes, outputfile):

    # keep input's file header
    h = readHeaderFromFile(inputfile)
    h["nz"] = len(indexes)
    h["mz"] = h["nz"]

    # write main header
    headerbytes = makeHeaderData(h)
    f = open(outputfile, "wb")
    f.write(headerbytes)

    sourcemrc = mmap(inputfile)

    for index in indexes:
        appendArray(sourcemrc[int(index), :, :], f)
    f.close()


def appendArray(a, f):
    """a = numpy array, f = open file object"""
    # make sure array is right type for MRC
    a = asMRCtype(a)

    # make sure array is in native byte order
    if not a.dtype.isnative:
        a = a.byteswap()

    # seek to end of file
    f.seek(0, 2)

    ## write data in smaller chunks.  Otherwise, writing from
    ## windows to a samba share will fail if image is too large.
    smallersize = 16 * 1024 * 1024
    b = a.ravel()
    items_per_write = int(smallersize / a.itemsize)
    for start in range(0, b.size, items_per_write):
        end = start + items_per_write
        b[start:end].tofile(f)


def append(a, filename):
    # read existing header
    f = open(filename, "rb+")
    f.seek(0)
    headerbytes = f.read(1024)
    oldheader = parseHeader(headerbytes)

    # make a header for new array
    newheader = {}
    updateHeaderUsingArray(newheader, a)

    ## check that new array is compatible with old array
    notmatch = []
    for key in ("nx", "ny", "mode"):
        if newheader[key] != oldheader[key]:
            notmatch.append(key)
    if notmatch:
        raise RuntimeError(
            "Array to append is not compatible with existing array: %s" % (notmatch,)
        )

    ## update old header for final MRC
    oldheader["nz"] += newheader["nz"]
    ## (could also update some other fields of header...)

    headerbytes = makeHeaderData(oldheader)
    f.seek(0)
    f.write(headerbytes)

    appendArray(a, f)

    f.close()


def read_(filename):
    """
Read the MRC file given by filename, return numpy ndarray object
	"""
    f = open(filename, "rb")
    headerbytes = f.read(1024)
    headerdict = parseHeader(headerbytes)
    a = readDataFromFile(f, headerdict)

    ## store keep header with image
    setHeader(a, headerdict)
    f.close()
    return a


"""
qh mrc reader

def parse_mrc(fname, lazy=False):
    # parse the header
    header = MRCHeader.parse(fname)
    
    ## get the number of bytes in extended header
    extbytes = header.fields['next']
    start = 1024+extbytes # start of image data

    dtype = DTYPE_FOR_MODE[header.fields['mode']]
    nz, ny, nx = header.fields['nz'], header.fields['ny'], header.fields['nx']
    
    # load all in one block
    if not lazy:
        with open(fname, 'rb') as fh:
            fh.read(start) # skip the header + extended header
            array = np.fromfile(fh, dtype=dtype).reshape((nz,ny,nx))

    # or list of LazyImages
    else:
        stride = dtype().itemsize*ny*nx
        array = [LazyImage(fname, (ny, nx), dtype, start+i*stride) for i in range(nz)]
    return array, header
"""


def read(filename):
    with open(filename, "rb") as f:
        headerbytes = f.read(1024)
        headerdict = parseHeader(headerbytes)
        dt = headerdict["dtype"]
        if dt == "float32":
            dt = numpy.dtype("f4")
        elif dt == "int16":
            dt = numpy.dtype("i2")
        elif dt == "uint16":
            dt = numpy.dtype("u2")
        elif dt == "int8":
            dt = numpy.dtype("i1")
        elif dt == "uint8":
            dt = numpy.dtype("u1")
        else:
            print(("ERROR - mrc type {} not recognized.".format(dt)))

        x = headerdict["nx"]
        y = headerdict["ny"]
        z = headerdict["nz"]

        total_header_size = 1024 + int(headerdict["nsymbt"])
        f.seek(total_header_size)
        if z > 1:
            image = numpy.reshape(numpy.fromfile(f, dt, y * x * z), [z, y, x]).astype(
                numpy.float32
            )
        else:
            image = numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x]).astype(
                numpy.float32
            )

        # print filename, dt, x, y, z, int(headerdict['nsymbt'])

    return image


def readframe(filename, frame):
    """
	Read a single frame from file.
	"""
    f = open(filename, "rb")
    headerbytes = f.read(1024)
    headerdict = parseHeader(headerbytes)
    dt = headerdict["dtype"]
    if dt == "float32":
        dt = numpy.dtype("f4")
    elif dt == "int16":
        dt = numpy.dtype("i2")
    elif dt == "uint16":
        dt = numpy.dtype("u2")
    elif dt == "int8":
        dt = numpy.dtype("i1")
    elif dt == "uint8":
        dt = numpy.dtype("u1")
    else:
        print(("ERROR - mrc type {} not recognized.".format(dt)))

    x = headerdict["nx"]
    y = headerdict["ny"]

    total_header_size = 1024 + int(headerdict["nsymbt"])
    f.seek(total_header_size + frame * y * x * dt.itemsize)
    image = numpy.reshape(numpy.fromfile(f, dt, y * x), [y, x]).astype(numpy.float32)
    f.close()
    return image


def setHeader(a, headerdict):
    """
Attach an MRC header to the array.
	"""
    weakattr.set(a, "mrcheader", headerdict)


def getHeader(a):
    """
Return the MRC header for the array, if it has one.
	"""
    return weakattr.get(a, "mrcheader")


def mmap(filename):
    """
Open filename as a memory mapped MRC file.  The returned object is
a numpy ndarray object wrapped around the memory mapped file.
	"""
    ## read only the header and parse it
    f = open(filename, "rb")
    headerbytes = f.read(1024)
    f.close()
    headerdict = parseHeader(headerbytes)

    ## open memory mapped file
    mrcdata = numpy.memmap(
        filename,
        dtype=headerdict["dtype"],
        mode="r",
        offset=1024,
        shape=headerdict["shape"],
        order="C",
    )
    ## attach header to the array
    setHeader(mrcdata, headerdict)
    return mrcdata


def readHeaderFromFile(filename):
    f = open(filename, "rb")
    h = f.read(1024)
    f.close()
    h = parseHeader(h)
    return h


def testHeader():
    infilename = sys.argv[1]
    f = open(infilename, "rb")
    h = f.read(1024)
    f.close()
    h = parseHeader(h)
    printHeader(h)


def testWrite():
    a = numpy.zeros((16, 16), numpy.float32)
    write(a, "a.mrc")


def testStack():
    ## write individual files
    files = []
    tilts = []
    for tilt in (1, 2, 3, 4, 5):
        a = tilt * numpy.ones((8, 8), numpy.float32)
        filename = "tilt%03d.mrc" % (tilt,)
        write(a, filename)
        files.append(filename)
        tilts.append(tilt)

    ## make stack
    outputname = "stack.mrc"
    stack(files, tilts, outputname)


if __name__ == "__main__":
    testHeader()
    # testWrite()
    # testStack()
