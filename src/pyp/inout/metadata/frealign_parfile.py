"""
Classes for reading/manipulating/writing FREALIGN parameter (.par, .parx, and .star) files (parfiles).

Provides string and header templates for four base formats and the corresponding extended formats.
1. cclin - 13 cols
2. new - 16 cols
3. frealignx - 17 cols
4. cistem2 - star file (NOTE: not fully supported)

User stories
------------
Completed
1. Users need to read from par file
2. Users need to write out to par file

TODO
2.1 Users can extend the parfile dynamically, e.g., from regular -metric new parfile to .parx
3. Users need to be able to append dynamically
4. Users can call a method to find out the number of particles
5. A method to get the indexes based on the particle number
6. Merge parameters from multiple files
7. isfrealignx, ismetricnew and such methods
8. Ability to convert between the formats
9. Users can generate an empty parfile -- potentially from other metadata
10. Take parameter entry corresponding to the first frame
"""

from inspect import Parameter
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.utils import get_relative_path
from pyp.system import mpi

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

psi_col = 1
theta_col = 2
phi_col = 3
shx_col = 4
shy_col = 5
score_col = 14

index_col = 1 - 1
film_col = 8 - 1
ptlind_col = 17 - 1
scanor_col = 20 - 1
parx_num_cols = 45
old_num_cols = 16

CCLIN = "cclin"
CC3M = "cc3m"
NEW = "new"
FREALIGNX = "frealignx"
CISTEM2 = "cistem2"

CCLIN_NUM_COL = 13
EXTENDED_CCLIN_NUM_COL = 13 + 29
CC3M_NUM_COL = CCLIN_NUM_COL
EXTENDED_CC3M_NUM_COL = EXTENDED_CCLIN_NUM_COL
NEW_NUM_COL = 16
EXTENDED_NEW_NUM_COL = 16 + 29
FREALIGNX_NUM_COL = 17
EXTENDED_FREALIGNX_NUM_COL = 17 + 29

CCLIN_PAR_STRING_TEMPLATE = (
    "%7d%8.2f%8.2f%8.2f%10.2f%10.2f%8.0f%6d%9.1f%9.1f%8.2f%8.2f%10d%11.4f%8.2f%8.2f"
)
EXTENDED_CCLIN_PAR_STRING_TEMPLATE = (
    CCLIN_PAR_STRING_TEMPLATE
    + "%9d%9.2f%9.2f%9d%9.2f%9.2f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f"
)
EXTENDED_CCLIN_PAR_STRING_TEMPLATE_WO_NO = EXTENDED_CCLIN_PAR_STRING_TEMPLATE[3:]

CC3M_PAR_STRING_TEMPLATE = CCLIN_PAR_STRING_TEMPLATE
EXTENDED_CC3M_PAR_STRING_TEMPLATE = EXTENDED_CCLIN_PAR_STRING_TEMPLATE

NEW_PAR_STRING_TEMPLATE = (
    "%7d%8.2f%8.2f%8.2f%10.2f%10.2f%8.0f%6d%9.1f%9.1f%8.2f%8.2f%10.0f%11.4f%8.2f%8.2f"
)
NEW_PAR_STRING_TEMPLATE_WO_NO = NEW_PAR_STRING_TEMPLATE[3:]

EXTENDED_NEW_PAR_STRING_TEMPLATE = EXTENDED_CCLIN_PAR_STRING_TEMPLATE
EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO = EXTENDED_NEW_PAR_STRING_TEMPLATE[3:]

FREALIGNX_PAR_STRING_TEMPLATE = "%7d%8.2f%8.2f%8.2f%10.2f%10.2f%8.0f%6d%9.1f%9.1f%8.2f%8.2f%8.2f%10.0f%11.4f%8.2f%8.2f"
EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE = "%7d%8.2f%8.2f%8.2f%10.2f%10.2f%8.0f%6d%9.1f%9.1f%8.2f%8.2f%8.2f%10.0f%11.4f%8.2f%8.2f%9d%9.2f%9.2f%9d%9.2f%9.2f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f%10.4f"
EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE_WO_NO = EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE[
    3:
]

CCLIN_PAR_HEADER = [
    """C FREALIGN CCLIN parameter file\n""",
    """C     1       2       3       4       5       6       7     8        9       10      11     12      13\n""",
    """C   NUM     PSI   THETA     PHI      SX      SY     MAG  FILM      DF1      DF2  ANGAST  PRESA  DPRESA\n""",
]

CC3M_PAR_HEADER = CCLIN_PAR_HEADER

EXTENDED_CCLIN_PAR_HEADER = [
    """C FREALIGN EXTENDED CCLIN parameter file\n""",
    """C     1       2       3       4       5       6       7     8        9       10      11     12      13         14      15      16       17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45\n""",
    """C   NUM     PSI   THETA     PHI      SX      SY     MAG  FILM      DF1      DF2  ANGAST  PRESA  DPRESA   PTLIND    TILTAN    DOSEXX    SCANOR    CNFDNC    PTLCCX      AXIS     NORM0     NORM1     NORM2  MATRIX00  MATRIX01  MATRIX02  MATRIX03  MATRIX04  MATRIX05  MATRIX06  MATRIX07  MATRIX08  MATRIX09  MATRIX10  MATRIX11  MATRIX12  MATRIX13  MATRIX14  MATRIX15      PPSI    PTHETA      PPHI\n""",
]

EXTENDED_CC3M_PAR_HEADER = EXTENDED_CCLIN_PAR_HEADER

EXTENDED_NEW_PAR_HEADER = [
    """C FREALIGN EXTENDED NEW parameter file\n""",
    """C     1       2       3       4         5         6       7     8        9       10      11      12        13         14      15      16       17        18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45\n""",
    """C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LOGP      SIGMA   SCORE  CHANGE   PTLIND    TILTAN    DOSEXX    SCANOR    CNFDNC    PTLCCX      AXIS     NORM0     NORM1     NORM2  MATRIX00  MATRIX01  MATRIX02  MATRIX03  MATRIX04  MATRIX05  MATRIX06  MATRIX07  MATRIX08  MATRIX09  MATRIX10  MATRIX11  MATRIX12  MATRIX13  MATRIX14  MATRIX15      PPSI    PTHETA      PPHI\n""",
]

EXTENDED_FREALIGNX_PAR_HEADER = [
    """C FREALIGN EXTENDED FREALIGNX parameter file\n""",
    """C     1       2       3       4         5         6       7     8        9       10      11      12      13        14         15      16      17       18        19        20        21        22        23        24        25        26        27        28        29        30        31        32        33        34        35        36        37        38        39        40        41        42        43        44        45        46\n""",
    """C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST  PSHIFT     OCC      LOGP      SIGMA   SCORE  CHANGE   PTLIND    TILTAN    DOSEXX    SCANOR    CNFDNC    PTLCCX      AXIS     NORM0     NORM1     NORM2  MATRIX00  MATRIX01  MATRIX02  MATRIX03  MATRIX04  MATRIX05  MATRIX06  MATRIX07  MATRIX08  MATRIX09  MATRIX10  MATRIX11  MATRIX12  MATRIX13  MATRIX14  MATRIX15      PPSI    PTHETA      PPHI\n""",
]

NEW_PAR_HEADER = [
    """C FREALIGN NEW parameter file\n""",
    """C     1       2       3       4         5         6       7     8        9       10      11      12        13         14      15      16\n""",
    """C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST     OCC      LOGP      SIGMA   SCORE  CHANGE\n""",
]

FREALIGNX_PAR_HEADER = [
    """C FREALIGNX parameter file\n""",
    """C     1       2       3       4         5         6       7     8        9       10      11      12      13        14         15      16      17\n""",
    """C    NO     PSI   THETA     PHI       SHX       SHY     MAG  FILM      DF1      DF2  ANGAST  PSHIFT     OCC      LOGP      SIGMA   SCORE  CHANGE\n""",
]


def getRangeFromTemplate(template, film_col):
    
    digits = template.replace('f','').replace('d', '').split('%')[1:]
    cumulative_digits = 0

    for idx, d in enumerate(digits):
        try:
            digit = int(d)
        except:
            digit = int(d.split('.')[0])

        if idx == 0:
            section1_start = digit + 1
        elif idx == film_col:
            section1_end = cumulative_digits
            section2_start = cumulative_digits + digit + 1

        cumulative_digits += digit
    
    section2_end = cumulative_digits

    return section1_start, section1_end, section2_start, section2_end



class ParameterEntry:
    def __init__(self, name, label="", format=""):
        self.name = name
        if len(label) > 0:
            self.label = label
        elif name in list(self.types.keys()):
            self.label = self.types[name][0]
        if len(format) > 0:
            self.format = format
        elif name in list(self.types.keys()):
            self.format = self.types[name][1]

    name = ""
    label = ""
    format = "%7d"
    data = ""

    types = {}
    types["PositionInStack"] = ["    POS", "%8d"]
    types["AnglePsi"] = ["     PSI", "%8.2f"]
    types["AngleTheta"] = ["   THETA", "%8.2f"]
    types["AnglePhi"] = ["     PHI", "%8.2f"]
    types["XShift"] = ["       SHX", "%10.2f"]
    types["YShift"] = ["       SHY", "%10.2f"]
    types["Defocus1"] = ["      DF1", "%9.1f"]
    types["Defocus2"] = ["      DF2", "%9.1f"]
    types["DefocusAngle"] = ["  ANGAST", "%8.2f"]
    types["PhaseShift"] = ["  PSHIFT", "%8.2f"]
    types["ImageActivity"] = ["  STAT", "%6d"]
    types["Occupancy"] = ["     OCC", "%8.2f"]
    types["LogP"] = ["      LogP", "%10d"]
    types["Sigma"] = ["      SIGMA", "%11.4f"]
    types["Score"] = ["   SCORE", "%8.2f"]
    types["ScoreChange"] = ["  CHANGE", "%8.2f"]
    types["PixelSize"] = ["    PSIZE", "%9.5f"]
    types["MicroscopeVoltagekV"] = ["    VOLT", "%8.2f"]
    types["MicroscopeCsMM"] = ["      Cs", "%8.2f"]
    types["AmplitudeContrast"] = ["    AmpC", "%8.2f"]
    types["BeamTiltX"] = ["  BTILTX", "%8.2f"]
    types["BeamTiltY"] = ["  BTILTY", "%8.2f"]
    types["ImageShiftX"] = ["  ISHFTX", "%8.2f"]
    types["ImageShiftY"] = ["  ISHFTY", "%8.2f"]
    types["Best2DClass"] = [" 2DCLS", "%6d"]
    types["BeamTiltGroup"] = ["  TGRP", "%6d"]
    types["StackFilename"] = [
        "                                      STACK_FILENAME",
        "%52s",
    ]
    types["OriginalImageFilename"] = [
        "                             ORIGINAL_IMAGE_FILENAME",
        "%52s",
    ]
    types["Reference3DFilename"] = [
        "                               REFERENCE_3D_FILENAME",
        "%52s",
    ]
    types["ParticleGroup"] = ["    PaGRP", "%9d"]
    types["PreExposure"] = ["  PREEXP", "%8.2f"]
    types["TotalExposure"] = ["  TOTEXP", "%8.2f"]


class Parameters:
    """Object to store complete data in a FREALIGN parameter file (parfile)."""

    versions = (CCLIN, CC3M, NEW, FREALIGNX, CISTEM2)

    def __init__(
        self, version, extended=False, data=None, prologue=None, epilogue=None
    ):
        """Constructs a Parameters object. Initalizies parfile type with version and extended.
        Any provided data, prologue and epilogue are also stored. 

        1. cclin - 13 cols
        2. cc3m - 13 cols
        3. new - 16 cols
        4. frealignx - 17 cols
        5. cistem2 - star file (NOTE: not fully supported)

        Parameters
        ----------
        version : str
            One of (CCLIN, CC3M, NEW, FREALIGNX, CISTEM2)
        extended : bool, optional
            Whether parfile is in extended format, by default False
        data : numpy.ndarray, optional
            Parameters in float, by default None
        prologue : list, optional
            Rows of strings representing comments before main data, by default None
        epilogue : list, optional
            Rows of strings representing comments after main data, by default None

        Raises
        ------
        ValueError
            If version not in list of acceptable versions
        """

        if version not in self.versions:
            raise ValueError(
                "Parameter version not in list of acceptable versions", self.versions
            )

        if not isinstance(data, np.ndarray) and not data:
            self.data = np.array([])
        self.data = np.array(data)
        self.extended = extended

        # attributes
        self.version = version
        self.columns = []

        # comments before and after main data
        if not prologue:
            self.prologue = []
        else:
            self.prologue = prologue
        if not epilogue:
            self.epilogue = []
        else:
            self.epilogue = epilogue

        # cistem_cols = ...

        if self.version == "cistem":

            self.columns.append(ParameterEntry("PositionInStack"))
            self.columns.append(ParameterEntry("AnglePsi"))
            self.columns.append(ParameterEntry("AngleTheta"))
            self.columns.append(ParameterEntry("AnglePhi"))
            self.columns.append(ParameterEntry("XShift"))
            self.columns.append(ParameterEntry("YShift"))
            self.columns.append(ParameterEntry("Defocus1"))
            self.columns.append(ParameterEntry("Defocus2"))
            self.columns.append(ParameterEntry("DefocusAngle"))
            self.columns.append(ParameterEntry("PhaseShift"))
            self.columns.append(ParameterEntry("ImageActivity"))
            self.columns.append(ParameterEntry("Occupancy"))
            self.columns.append(ParameterEntry("LogP"))
            self.columns.append(ParameterEntry("Sigma"))
            self.columns.append(ParameterEntry("Score"))
            self.columns.append(ParameterEntry("PixelSize"))
            self.columns.append(ParameterEntry("MicroscopeVoltagekV"))
            self.columns.append(ParameterEntry("MicroscopeCsMM"))
            self.columns.append(ParameterEntry("AmplitudeConstrast"))
            self.columns.append(ParameterEntry("BeamTiltX"))
            self.columns.append(ParameterEntry("BeamTiltY"))
            self.columns.append(ParameterEntry("ImageShiftX"))
            self.columns.append(ParameterEntry("ImageShiftY"))

    # save database to .star file
    def write_star(self, name):

        with open(name + ".star", "w") as f:

            f.write("# Written by pyp\n\n")
            f.write("data_\n\n")
            f.write("loop_\n")

            column = 1
            labels = "#"
            line = ""
            format = ""
            # use helper function to get particle format
            for entry in self.columns:
                f.write("_cisTEM" + entry.name + " #%d\n" % column)
                if "s" in entry.format:
                    labels += entry.format % entry.label
                else:
                    labels += entry.label
                format += entry.format
                column += 1
            labels += "\n"
            format += "\n"

            # write labels
            f.write(labels)

            for i in range(self.data.shape[0]):
                if not self.data.dtype.char == "S":
                    f.write(format % tuple(self.data[i, :column]))
                else:
                    line = ""
                    column = 0
                    for entry in self.columns:
                        if "s" in entry.format:
                            if "''" == self.data[i, column]:
                                line += entry.format % (self.data[i, column])
                            else:
                                line += entry.format % (
                                    "'" + self.data[i, column] + "'"
                                )
                        else:
                            line += entry.format % float(self.data[i, column])
                        column += 1
                    f.write(line + "\n")

    def get_string_template(self):
        """Returns correct par string template based on self.version and self.extended"""
        if self.version == FREALIGNX:
            if self.extended:
                return EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE
            return FREALIGNX_PAR_STRING_TEMPLATE
        if self.version == NEW:
            if self.extended:
                return EXTENDED_NEW_PAR_STRING_TEMPLATE
            return NEW_PAR_STRING_TEMPLATE
        if self.version == CCLIN:
            if self.extended:
                return EXTENDED_CCLIN_PAR_STRING_TEMPLATE
            return CCLIN_PAR_STRING_TEMPLATE
        if self.version == CC3M:
            if self.extended:
                return EXTENDED_CC3M_PAR_STRING_TEMPLATE
            return CC3M_PAR_STRING_TEMPLATE

    def get_string(self, prologue=True, epilogue=True):
        """Returns a string to be written as a parfile.
        
        Contains the prologue followed by data and finally the epilogue.
        
        Parameters
        ----------
        prologue : bool, optional
            Whether to include prologue, by default True
        epilogue : bool, optional
            Whether to include epilogue, by default True

        Returns
        -------
        str
            String representation of parfile
        """
        if prologue:
            lines = self.prologue.copy()
        else:
            lines = []
        particles = self.data.shape[0]
        for i in range(particles):
            output = self.get_string_template() % tuple(self.data[i])
            # if "frealignx" in self.version:
            #     output = FREALIGNX_PAR_STRING_TEMPLATE % tuple(self.data[i, :17])
            # elif "new" in self.version:
            #     output = NEW_PAR_STRING_TEMPLATE % tuple(self.data[i, :16])
            # else:
            #     output = CCLIN_PAR_STRING_TEMPLATE % tuple(self.data[i, :13])
            lines.append(output)
        if epilogue:
            lines.extend(self.epilogue.copy())

        return "".join([f"{line}\n" for line in lines])

    def __str__(self):
        """String representation of parameter file."""
        return self.get_string()

    def __repr__(self):
        return self.get_string()

    def write_file(self, filename: os.PathLike, **kwargs):
        """Write out parameter file

        Parameters
        ----------
        filename : os.PathLike
            Output parfile
        """
        with open(filename, "w") as f:
            f.writelines(self.get_string(**kwargs))

    def write_par(self, name):
        pass

    def write_parx(self, name):
        pass

    def read_star(self, name):
        pass

    # read database from .par file
    def read_par(self, name, mp):
        """Reads PYP parameters from par file

        Parameters
        ----------
        name : str
            filename excluding .par extension
        mp : dict
            PYP configuration
        """

        # TODO: update this method so that it replaces the other attributes
        par = self.from_file(name + ".par").data

        # construct new star data
        self.data = np.zeros([par.shape[0], 23])

        # POS, PSI, THETA, PHI, SHX, SHY
        self.data[:, :6] = par[:, :6]

        # DF1, DF2, ANGAST
        self.data[:, 6:9] = par[:, 8:11]

        # PSHIFT
        self.data[:, 9] = 0

        # STAT
        self.data[:, 10] = 1

        # OCC, LogP, SIGMA, SCORE
        self.data[:, 11:15] = par[:, 11:15]

        # PSIZE
        self.data[:, 15] = str(
            float(mp["scope_pixel"]) * float(mp["data_bin"]) * float(mp["extract_bin"])
        )

        # VOLT
        self.data[:, 16] = mp["scope_voltage"]

        # CS
        self.data[:, 17] = mp["scope_cs"]

        # AmpC
        self.data[:, 18] = mp["scope_wgh"]

        # BTILTX, BTILTY, ISHFTX, ISHFTY
        self.data[:, 19:] = 0

    # add column to database
    def add_column(self, entry, value):

        exists = False
        count = 0
        for i in self.columns:
            if entry.name == i.name:
                exists = True
                break
            count += 1

        if not exists:
            if "s" in entry.format:
                field_size = len(value) + 3
                entry.format = "%{0}s".format(field_size)

            self.columns.append(entry)
            self.data = np.hstack((self.data, np.full((self.data.shape[0], 1), value)))
        else:
            # change array type if new data doesn't fit
            # TODO: string representation -> integer representation?
            if len(value) > self.data.dtype.itemsize:
                self.data = self.data.astype("S" + str(len(value) + 1))
            self.data[:, count] = value

            field_size = len(value) + 3
            if len(self.columns[count].label) < field_size:
                self.columns[count].label = (
                    "%{}s".format(field_size) % self.columns[count].label
                )
                self.columns[count].format = "%{0}s".format(field_size)

    def read_star(self, name):

        fields = [
            line.split()[0].replace("_cisTEM", "")
            for line in open(name + ".star")
            if line.startswith("_cisTEM")
        ]

        # labels = [ line.split()[1:] for line in open( name + '.star' ) if line.startswith('#    POS') ][0]

        self.columns = []
        for f in fields:
            self.columns.append(ParameterEntry(f))

        self.data = np.array(
            [
                line.split()
                for line in open(name + ".star")
                if not line.startswith("_")
                and not line.startswith("#")
                and not line.startswith("data_")
                and not line.startswith("loop_")
                and len(line) > 2
            ],
            dtype="str",
        )

    # compress
    @staticmethod
    def compress_parameter_file(input, output, threads=1):
        """Compress parameter file

        Parameters
        ----------
        input : string
            Input file
        output : string
            Output file
        threads: int
            Number of parallel threads
        """
        # thread greater than 10 is actually slower
        assert "/" not in input, f"Input(s) to pbzip2 should only contain basename, please go to the file directory before compressing"

        threads = 7 if threads > 7 else threads
        command = f"tar -v -c -h -v {input} | pbzip2 -c -v -p{threads} > {output}"
        run_shell_command(command, verbose=False)

    # compress
    @staticmethod
    def decompress_file(input, threads=1):
        """Compress file(s)

        Parameters
        ----------
        input : string
            Input file(s)
        threads : int
            Number of parallel threads
        """
        # thread greater than 10 is actually slower
        threads = 7 if threads > 7 else threads
        
        command = f"pbzip2 -v -d -c -p{threads} {input} | tar x"
        run_shell_command(command, verbose=False)

    # decompress paramater file in scratch directory
    @staticmethod
    def decompress_parameter_file(input, threads=1):
        """Decompress parameter file

        Parameters
        ----------
        input : string
            Input file
        threads: int
            Number of parallel threads
        """

        if os.path.exists(input + ".bz2"):
            input += ".bz2"
        if input.endswith(".bz2"):
            # decompress file in local scratch directory
            folder = Path(input).absolute().parent
            current_dir = os.getcwd()
            try:
                # try to go to local scratch first
                os.chdir(os.environ["PYP_SCRATCH"])
            except: 
                # if we don't have permission or it does not exist, 
                # try to go to the folder where the input is 
                os.chdir(folder)
            Parameters.decompress_file(input, threads)
            output = os.path.join(os.getcwd(), Path(input).name[:-4])
            os.chdir(current_dir)
        else:
            output = input
        return output
    
    @staticmethod
    def decompress_parameter_file_and_move(file: Path, new_file: Path, micrograph_list: list = [], threads=1):
        # delete the file if it already exists in the new path
        if new_file.exists(): 
            shutil.rmtree(new_file)
        
        # decompress the file into folder
        assert (str(file).endswith(".bz2")), f"{file} needs to be compressed in .bz2 format."
        decompressed_file = Parameters.decompress_parameter_file(str(file), threads)
        assert (os.path.isdir(decompressed_file)), f"{file} is not a folder after decompression."

        # check if the folder contains all the parameter files
        if len(micrograph_list) > 0:
            for micrograph in micrograph_list:
                assert ((Path(decompressed_file) / f"{micrograph}.cistem").exists()), f"{micrograph}.cistem is not in {file}."
                assert ((Path(decompressed_file) / f"{micrograph}_extended.cistem").exists()), f"{micrograph}_extended.cistem is not in {file}."

        shutil.move(decompressed_file, new_file)

    @staticmethod
    def write_parameter_file(output_fname, contents, parx=False, frealignx=False):
        """Write out parameter file with contents in a numpy array.

        Parameters
        ----------
        output_fname : PathLike
            Output parameter filename
        contents : numpy.ndarray
            Parameter contents
        parx : bool, optional
            Whether to use extended parameter file type, by default False
        """
        allparxs = []
        if not frealignx:
            template, header = (
                (EXTENDED_NEW_PAR_STRING_TEMPLATE, EXTENDED_NEW_PAR_HEADER)
                if parx
                else (NEW_PAR_STRING_TEMPLATE, NEW_PAR_HEADER)
            )
        else:
            template, header = (
                (EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE, EXTENDED_FREALIGNX_PAR_HEADER)
                if parx
                else (FREALIGNX_PAR_STRING_TEMPLATE, FREALIGNX_PAR_HEADER)
            )
        for row in contents:
            allparxs.append(template % tuple(row.tolist()))

        f = open(output_fname, "w")
        logger.info(f"Writing parfile {output_fname}")
        f.writelines(header)

        f.writelines("{}\n".format(item) for item in allparxs)
        f.close()

    @classmethod
    def take_first_parameter(cls, input_fname, output_fname, parx=False):
        """Take the parameter entry corresponding to the first frame."""
        input_arr = cls.from_file(input_fname).data

        allparxs = []

        template, header = (
            (EXTENDED_NEW_PAR_STRING_TEMPLATE, EXTENDED_NEW_PAR_HEADER)
            if parx
            else (NEW_PAR_STRING_TEMPLATE, NEW_PAR_HEADER)
        )

        allparxs.append(template % tuple(input_arr[0].tolist()))

        f = open(output_fname, "w")

        f.writelines(header)

        f.writelines("{}\n".format(item) for item in allparxs)
        f.close()

    @classmethod
    def fake_scores(cls, filename, first_score=1.0, rest_score=2.0):
        """Write fake scores in parfile to allow for reconstruction."""
        params = cls.from_file(filename).data
        params[:, score_col] = rest_score
        params[0, score_col] = first_score
        cls.write_parameter_file(filename, params)


    @staticmethod
    def merge_parameters(
        inputlist,
        filename,
        metric,
        update_film=False,
        start_film_idx=0,
        parx=False,
        frealignx=False,
    ):
        """Merges a list of .par files and outputs to filename."""
        if "cc" in metric:
            if parx:
                template, header = (
                    EXTENDED_CCLIN_PAR_STRING_TEMPLATE,
                    EXTENDED_CCLIN_PAR_HEADER,
                )
            else:
                template, header = (
                    CCLIN_PAR_STRING_TEMPLATE,
                    CCLIN_PAR_HEADER,
                )

        elif frealignx:
            if parx:
                template, header = (
                    EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE,
                    EXTENDED_FREALIGNX_PAR_HEADER,
                )
            else:
                template, header = (
                    FREALIGNX_PAR_STRING_TEMPLATE,
                    FREALIGNX_PAR_HEADER,
                )
        else: # not FREALIGNX OR CCLIN then NEW
            if parx:
                template, header = (
                    EXTENDED_NEW_PAR_STRING_TEMPLATE,
                    EXTENDED_NEW_PAR_HEADER,
                )
            else:
                template, header = (
                    NEW_PAR_STRING_TEMPLATE,
                    NEW_PAR_HEADER,
                )
        # else:
        #     template, header = (NEW_PAR_STRING_TEMPLATE, NEW_PAR_HEADER)


        # if only one file is present
        if len(inputlist) == 1:
            shutil.copy2(inputlist[0], filename)
            return

        index_fmt = "%" + template.split("%")[1] 
        film_fmt = "%" + template.split("%")[film_col+1]

        section1_start, section1_end, section2_start, section2_end = getRangeFromTemplate(template, film_col)

        commands = []
        # create new parfile without having header, footer
        # trim the two sections (index - film, film - end) by character position 
        for idx, parfile in enumerate(inputlist):
            commands.append("grep '^[^C]' %s > %s && cut -c%d-%d %s > %s && cut -c%d-%d %s > %s" % (parfile, parfile + '.tmp',
                                                                                                    section1_start, section1_end, parfile + '.tmp', f"{idx}_1.tmp",
                                                                                                    section2_start, section2_end, parfile + '.tmp', f"{idx}_2.tmp"))
        mpi.submit_jobs_to_workers(commands, os.getcwd(), silent=True)

        # update (write) INDEX, FILM columns for each parfile
        if frealignx:
            cur_film = -1
        else:
            cur_film = 0

        cur_index = 0
        for idx, parfile in enumerate(inputlist):

            # read only film column from files 
            films_df = pd.read_csv(parfile + '.tmp', header=None, delim_whitespace=True, usecols=[film_col])
            num_rows = films_df.shape[0]

            films_arr = films_df.to_numpy()
            if update_film:
                films_arr[:, 0] += cur_film
            np.savetxt('films.tmp', films_arr, fmt=film_fmt)
            if frealignx:
                cur_film = films_arr[num_rows-1,0]
            else:
                cur_film = films_arr[num_rows-1,0] + 1

            # write corresponding first column
            indexes_arr = np.mod(np.arange(cur_index+1, cur_index+num_rows+1, 1), int(1e7))
            np.savetxt('indexes.tmp', indexes_arr, fmt=index_fmt)
            cur_index += num_rows

            # concatenate first column, film column with the rest of parfiles
            command = "paste -d '' {0} > {1}".format(" ".join([ 'indexes.tmp', f"{idx}_1.tmp", 'films.tmp', f"{idx}_2.tmp"]), 
                                                    parfile)
            run_shell_command(command, verbose=False)

            # remove tmp files
            os.remove('films.tmp')
            os.remove('indexes.tmp')
            os.remove(parfile + '.tmp')

        # stack all the updated parfile together into one
        with open(filename, "w") as f:
            f.writelines(header)
        # command = "cat {0} >> {1}".format(" ".join(inputlist), filename)
        if len(inputlist) > 500:
            splits = [inputlist[i:i+500] for i in range(0, len(inputlist), 500)]
            for batch in splits:
                command = "cat {0} >> {1}".format(" ".join(batch), filename)
                run_shell_command(command, verbose=False)
        else:
            command = "cat {0} >> {1}".format(" ".join(inputlist), filename)
            run_shell_command(command, verbose=False)
        
        [os.remove(f) for f in os.listdir(".") if f.endswith(".tmp")]
        
        """
        input_arr = [
            pd.DataFrame(
                np.loadtxt(parfile, ndmin=2, comments="C", dtype=float)
                # [line.split() for line in open(parfile) if not line.startswith("C")],
                # dtype="float",
            )
            for parfile in inputlist
        ]
        grouped_arr = [
            [np.array(film, ndmin=2) for _, film in f.groupby([film_col])] for f in input_arr
        ]

        # allocate final array in memory first 
        row = sum([arr.shape[0] for arr in input_arr])
        col = input_arr[0].shape[1]
        output_arr = np.zeros((row, col))

        row_idx = 0
        film_idx = start_film_idx
        for f in grouped_arr:
            for film in f:
                number_rows_local = film.shape[0]
                
                # update film index (FILM column)
                film[:, film_col] = film_idx

                # add current block to our final array pre-allocated in the memory 
                output_arr[row_idx: row_idx+number_rows_local, :] = film

                row_idx += number_rows_local
                film_idx += 1
        
        # update the image sequence (starting from 1)
        output_arr[:, 0] = [(i+1) % int(1e7) for i in range(row)]

        np.savetxt( filename, 
                    output_arr, 
                    fmt=template, 
                    header="\n".join([_.strip() for _ in header]), 
                    comments=""
                  )
        
        """
        """
        row_idx = 1
        film_idx = start_film_idx
        for f in grouped_arr:
            # update the columns
            for film in f:
                for row in film:
                    # limit indexes to 8 columns to comply with .par format restrictions
                    row[index_col] = row_idx % int(1e7)
                    if update_film:
                        row[film_col] = film_idx
                    row_idx += 1
                    # logger.info("film idx %d", row[film_col])
                film_idx += 1

        output_arr = np.vstack([np.vstack(f) for f in grouped_arr])
        
        allparxs = []

        for row in output_arr:
            allparxs.append(template % tuple(row.tolist()))

        with open(filename, "w") as f:
            f.writelines(header)
            f.writelines("{}\n".format(item) for item in allparxs)
        """
    @staticmethod
    def csp_merge_parameters(inputlist, filename, frealignx):
        """Merges multiple FREALIGN extended parfiles (.parx) from CSP swarm functions.

        Parameters
        ----------
        inputlist : List[str]
            List of micrographs
        filename : str
            Output filename
        """

        parxlist = ["frealign/" + line for line in inputlist]

        with open(filename, "w") as f:

            if frealignx:
                f.writelines(EXTENDED_FREALIGNX_PAR_HEADER)
                lcolumn = 52
                hcolumn = 58
            else:
                f.writelines(EXTENDED_NEW_PAR_HEADER)
                lcolumn = 52
                hcolumn = 58

            # concatenate all files
            counter = 1
            film = 1
            for parx in parxlist:
                if os.path.exists(parx):
                    with open(parx, "r") as infile:
                        for line in infile:
                            # update film number
                            f.write(
                                "%7d" % (counter)
                                + line[:lcolumn]
                                + "%6d" % (film - 1)
                                + line[hcolumn:]
                            )
                            counter += 1
                    film += 1
                else:
                    logger.error("{0} does not exist.".format(parx))

    @staticmethod
    def csp_merge_parameters_frame_refinement(inputlist, filename, p_object, regions):
        """Merge multiple splitted parx files generated by CSPT and remove duplicate particles (in overlapped regions) by accumulating their alignment parameters 

        Parameters
        ----------
        inputlist : list[str]
            List that has the name of splitted parfiles 
        filename : str
            The name of the merged parx file
        p_object : Parameters
            Frealign parameter object 

        Returns
        ----------
        numpy array :
            Parx file metadata
        """

        # concatenate all files (assume they are in the same tilt-series/movie)
        # Therefore, no need to update film
        # par_array_list = [Parameters.from_file(parx).data for parx in inputlist]
        par_array_list = []
        for parx in inputlist:
            try:
                par_array_list.append(Parameters.from_file(parx).data)
            except UnboundLocalError:
                logger.error(f"{parx} does not contain anything.")
                continue

        merge_array = np.concatenate(par_array_list, axis=0)

        # IMPORTANT: sort line based on the image index
        merge_array = merge_array[np.argsort(merge_array[:, 0])]

        if not regions:
            return merge_array
        else:
            # merge alignment parameters (rotations and translations) for duplicates
            merge_no_duplicates = merge_array[
                np.unique(merge_array[:, 0].astype("int"), return_index=True)[1], :
            ]
            """
            for idx, _line in enumerate(merge_no_duplicates):
                duplicates = np.atleast_2d(merge_array[merge_array[:, 0] == _line[0]])[
                    :, 1:6
                ]
                sum_alignments = np.sum(duplicates, axis=0) / duplicates.shape[0]

                merge_no_duplicates[idx, 1:6] = sum_alignments[:]
            """
            # return array instead of writing file
            if merge_array.shape[0] > 0:

                # have to cleanup splitted parfiles, otherwise they will be merged in the next CSP iteration
                [os.remove(parx) for parx in inputlist]
                return merge_no_duplicates

    
    
    @staticmethod 
    def addFrameIndexInScanord(input_parfile, output_parfile, forward=True):
        
        """ Modify SCANORD column considering tilts + frames or undo the modification (only needed by tomo)

        Parameters
        ----------
        input_parfile : String
            Relative path to input parfile
        output_parfile : String
            Relative path to output parfile
        forward : bool
            Add granularity of frames into scanord (forward or backward)
        """

        SCANORD_COL = 20 - 1
        CNFDNC_COL = 21 - 1
        PTLIIND_COL = 17 - 1

        input_object = Parameters.from_file(input_parfile)
        input_data = input_object.data
        
        num_frames = len(np.unique(input_data[:, CNFDNC_COL]).astype(int))
        num_tilts = len(np.unique(input_data[:, SCANORD_COL]).astype(int))
        max_scanord = max(np.unique(input_data[:, SCANORD_COL]).astype(int))

        pair = input_data[:, [PTLIIND_COL, SCANORD_COL]]
        unique_pair = np.unique(pair, axis=0)

        if forward:
            logger.info("Add frame index from CNFDNC to SCANORD column")
            input_data[:, SCANORD_COL] = input_data[:, SCANORD_COL] * num_frames + input_data[:, CNFDNC_COL]
        # elif not forward and num_tilts > 100 or max_scanord > 100: 
        elif pair.shape[0] == unique_pair.shape[0]:    
            logger.info("Undo frame index from SCANORD column") 
            input_data[:, SCANORD_COL] = (input_data[:, SCANORD_COL] - input_data[:, CNFDNC_COL]) / num_frames
        else:
            return 
        input_object.data = input_data
        input_object.write_file(output_parfile)


    @staticmethod 
    def populateFrameIndexInScanord(input_parfile, output_parfile, forward=True):

        """ Modify SCANORD column to having either all 0s or frame indexes (only needed by spr)

        Parameters
        ----------
        input_parfile : String
            Relative path to input parfile
        output_parfile : String
            Relative path to output parfile
        forward : bool
            Different ways to deal with SCANORD column
        """

        SCANORD_COL = 20 - 1
        CNFDNC_COL = 21 - 1

        input_object = Parameters.from_file(input_parfile)
        input_data = input_object.data

        if forward:
            logger.info("Copy frame index from CNFDNC to SCANORD column")
            input_data[:, SCANORD_COL] = input_data[:, CNFDNC_COL]
        else:
            logger.info("Set all values in SCANORD to zero")
            input_data[:, SCANORD_COL] = 0

        input_object.data = input_data
        input_object.write_file(output_parfile)


    @staticmethod
    def removeFramesFromParFile(par_object, parameters):
        """ Convert parfile with frames to shorter one without frames, in order to run normal refine3d (now only support spr)

        Parameters
        ----------
        par_object : ParFile
            Parfile object defined in frealign_parfile
        parameters : Dict
            PYP config parameters

        """
        PSI_COL = 2 - 1
        THETA_COL = 3 - 1 
        PHI_COL = 4 - 1 
        SHIFTX_COL = 5 - 1
        SHIFTY_COL = 6 - 1

        FRAME_COL = 21 - 1
        MICROGRAPH_X_COL = 39 - 1
        MICROGRAPH_Y_COL = 40 - 1 
        FRAME_X_COL = 41 - 1
        FRAME_Y_COL = 42 - 1
        PPSI_COL = 43 - 1
        PTHETA_COL = 44 - 1 
        PPHI_COL = 45 - 1 

        logger.info("Converting parfile with frames to w/o frames before running refine3d refinement")
        par_data = par_object.data
        
        # sanity check if input parfile really contains frame info (frame col should have more than one unique number) 
        frame_idx = np.unique(par_data[:, FRAME_COL])
        assert (len(frame_idx) > 1), f"Input parfile should contain frames, but only one frame detected" 

        # remove lines where frame idx is not 0
        par_data = par_data[par_data[:, FRAME_COL] == 0]
        
        # copy particle rotation & translation from extended part to the first 5 columns
        par_data[:, PSI_COL:PHI_COL+1] = -par_data[:, PPSI_COL:PPHI_COL+1]
        par_data[:, SHIFTX_COL:SHIFTY_COL+1] = par_data[:, MICROGRAPH_X_COL:MICROGRAPH_Y_COL+1]

        # renumber the index from 1
        par_data[:, 0] = np.array([_+1 for _ in range(par_data.shape[0])])

        par_object.data = par_data

        return par_object


    @staticmethod 
    def copyAlignmentToParFileWithFrames(parfile_frames_data, parfile_wo_frames):
        """ Copy the refined alignments from shorter parfile w/o frames to longer parfile with frames

        Parameters
        ----------
        parfile_frames_data : Numpy array
            Parfile metadata in 2D array 
        parfile_wo_frames : String
            Path to parfile without frames (typically generated by refine3d)
        parameters : Dict
            PYP config parameters
        
        Returns
        ----------
        Numpy array (2D)
            Updated metadata 
        """
        PSI_COL = 2 - 1
        THETA_COL = 3 - 1 
        PHI_COL = 4 - 1 
        SHIFTX_COL = 5 - 1
        SHIFTY_COL = 6 - 1

        SCORE_COL = 15 - 1
        PTLIND_COL = 17 - 1
        FRAME_COL = 21 - 1
        MICROGRAPH_X_COL = 39 - 1
        MICROGRAPH_Y_COL = 40 - 1 
        FRAME_X_COL = 41 - 1
        FRAME_Y_COL = 42 - 1
        PPSI_COL = 43 - 1
        PTHETA_COL = 44 - 1 
        PPHI_COL = 45 - 1 

        par_data_frames = parfile_frames_data
        par_data = Parameters.from_file(parfile_wo_frames).data

        # copy particle alignment to its longer counterpart
        for parline in par_data:
            current_ptlind = parline[PTLIND_COL]
            loc_parfile_frames = np.where(par_data_frames[:, PTLIND_COL].astype(int) == current_ptlind)[0]
            num_frames = len(loc_parfile_frames)
            
            # update shift
            new_par_shifts = np.tile(parline[SHIFTX_COL:SHIFTY_COL+1], (num_frames, 1)) 
            par_data_frames[loc_parfile_frames, SHIFTX_COL:SHIFTY_COL+1] += (new_par_shifts - par_data_frames[loc_parfile_frames, MICROGRAPH_X_COL:MICROGRAPH_Y_COL+1])
            par_data_frames[loc_parfile_frames, MICROGRAPH_X_COL:MICROGRAPH_Y_COL+1] = new_par_shifts
            
            # update rotation 
            new_par_rot = np.tile(parline[PSI_COL:PHI_COL+1], (num_frames, 1)) 
            par_data_frames[loc_parfile_frames, PPSI_COL:PPHI_COL+1] = new_par_rot * -1
            par_data_frames[loc_parfile_frames, PSI_COL:PHI_COL+1] = new_par_rot

            # update score
            par_data_frames[loc_parfile_frames, SCORE_COL] = parline[SCORE_COL] 

        
        return par_data_frames



    @staticmethod
    def extendParFileWithFrames(
        par_object, allboxes, xf_frames, parameters, scanords=[0]
    ):
        """Convert short parxfile/allboxes file to longer ones that contain frame information for CSP frame refinement 

        Parameters
        ----------
        par_object : ParFile
            Parfile object defined in frealign_parfile
        allboxes : 2D np array
            array containing 2D coordiantes
        xf_frames : 3D np array
            array containing frame transformations per tilt 
        scanords : np array
            array containing scanning order 

        Returns
        -------
            2 lists 
            allboxes and allparxs 
        """

        PSI_COL = 2 - 1
        THETA_COL = 3 - 1 
        PHI_COL = 4 - 1 
        SHIFTX_COL = 5 - 1
        SHIFTY_COL = 6 - 1
        OCC_COL = 12 - 1
        SCANORD_COL = 20 - 1
        FRAME_COL = 21 - 1
        MICROGRAPH_X_COL = 39 - 1
        MICROGRAPH_Y_COL = 40 - 1 
        FRAME_X_COL = 41 - 1
        FRAME_Y_COL = 42 - 1
        PPSI_COL = 43 - 1
        PTHETA_COL = 44 - 1 
        PPHI_COL = 45 - 1 

        # figure out the number of frames for each scanning order (in case frame number varies)
        num_frames_scanord = {}
        max_num_frames = 0
        for idx, scanord in enumerate(scanords):
            num_frames_scanord[scanord] = len(xf_frames[idx])
            if len(xf_frames[idx]) > max_num_frames:
                max_num_frames = len(xf_frames[idx])

        # First, make up a very big 2D array to speed up expanding operation
        parx_frame = np.zeros(
            (max_num_frames * par_object.data.shape[0], par_object.data.shape[1])
        )
        allboxes_frame = np.zeros(
            (max_num_frames * len(allboxes), len(allboxes[0]) + 1)
        )

        # go through each line in short parfile and duplicate it by the number of frames
        curr = 0
        for idx, line in enumerate(par_object.data):

            num_frames = num_frames_scanord[line[SCANORD_COL]]
            frame_idxs = np.reshape(
                np.array([i for i in range(num_frames)]), (1, num_frames)
            )

            parlines = np.tile(line, (num_frames, 1))

            if "spr" in parameters["data_mode"].lower():
                
                # parlines[:, SCANORD_COL] = frame_idxs[0]
                # parlines[:, FRAME_COL] = 0.0
                parlines[:, SCANORD_COL] = 0
                parlines[:, FRAME_COL] = frame_idxs[0]

                # preserve original shift x/y (obtained using normal refine3d) 
                parlines[:, MICROGRAPH_X_COL] = parlines[:, SHIFTX_COL] 
                parlines[:, MICROGRAPH_Y_COL] = parlines[:, SHIFTY_COL] 
                
                # preserve original rotation (obtained using normal refine3d) 
                parlines[:, PPSI_COL] = - parlines[:, PSI_COL]
                parlines[:, PTHETA_COL] = - parlines[:, THETA_COL]
                parlines[:, PPHI_COL] = - parlines[:, PHI_COL]
                
                parlines[:, FRAME_X_COL] = 0.0 
                parlines[:, FRAME_Y_COL] = 0.0

                
            else:
                 
                parlines[:, FRAME_COL] = frame_idxs[0]
                
                x_err = xf_frames[scanords.index(line[SCANORD_COL])][:, 4] - np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 4])
                y_err = xf_frames[scanords.index(line[SCANORD_COL])][:, 5] - np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 5])
                
                parlines[:, SHIFTX_COL] -= x_err * parameters["scope_pixel"]
                parlines[:, SHIFTY_COL] -= y_err * parameters["scope_pixel"]
                parlines[:, FRAME_X_COL] -= x_err * parameters["scope_pixel"]
                parlines[:, FRAME_Y_COL] -= y_err * parameters["scope_pixel"]
                
                """
                parlines[:, FRAME_X_COL] = parlines[:, FRAME_X_COL] - (
                    np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 4])
                    * parameters["scope_pixel"]
                )
                parlines[:, FRAME_Y_COL] = parlines[:, FRAME_Y_COL] - (
                    np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 5])
                    * parameters["scope_pixel"]
                )
                """

            boxes = np.tile(allboxes[idx], (num_frames, 1))
            boxes = np.hstack((boxes, frame_idxs.reshape(frame_idxs.shape[1], 1)))
            
            boxes[:, 0] = boxes[:, 0] - ( 
                    np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 4])
                )
            boxes[:, 1] = boxes[:, 1] - ( 
                    np.round_(xf_frames[scanords.index(line[SCANORD_COL])][:, 5])
                )


            parx_frame[curr : curr + num_frames, :] = parlines
            allboxes_frame[curr : curr + num_frames, :] = boxes

            curr += num_frames

        # trim the rest of unused lines
        parx_frame = parx_frame[:curr, :]
        allboxes_frame = allboxes_frame[:curr, :]

        # convert them to list
        allboxes_frame = allboxes_frame.astype(int).tolist()

        allparxs_frame = [[]]
        
        for i in range(parx_frame.shape[0]):
            line = EXTENDED_NEW_PAR_STRING_TEMPLATE_WO_NO % tuple(parx_frame[i, 1:])
            allparxs_frame[0].append(line)

        return allboxes_frame, allparxs_frame

    @classmethod
    def generateParameterFiles(cls, inputlist, dataset, name, astigmatism="False"):
        """Generate both Frealign and Relion style parameter files"""
        # if not os.path.isfile( 'frealign/{0}_01.par'.format(name) ):
        cls.generateFrealignParFile(inputlist, dataset, name, astigmatism)
        # if not os.path.isfile( 'relion/{0}.star'.format(name) ):
        cls.generateRelionParFileNew(name)

    @classmethod
    def generateFrealignParFile(
        cls,
        inputlist,
        dataset,
        name,
        astigmatism="False",
        local="False",
        extract_cls=0,
    ):

        defocuslist = ["ctf/" + line.strip() + ".ctf" for line in inputlist]
        boxxlist = ["box/" + line.strip() + ".boxx" for line in inputlist]
        ctflist = ["ctf/" + line.strip() + "_ctf.txt" for line in inputlist]

        f = open("frealign/{0}_01.par".format(name), "w")

        f.writelines(NEW_PAR_HEADER)
        count = 1
        film = 0

        # for sname, stack, defocus in zip( inputlist, stacklist, defocuslist ):
        for defocus, boxxs, lctf in zip(defocuslist, boxxlist, ctflist):

            if os.path.exists(boxxs):
                ctf = np.loadtxt(defocus)
                boxx = np.loadtxt(boxxs)
                index_in_film = 0
                if ctf.shape[0] < 12:
                    logger.error(
                        "Not enough parameters in %s. Re-run ctf estimation." % defocus
                    )
                    return
                number_of_particles = np.sum(
                    np.logical_and(boxx[:, 4] == 1, boxx[:, 5] >= int(extract_cls))
                )

                for p in range(number_of_particles):
                    if local and os.path.exists(lctf):

                        ctf_local = np.loadtxt(lctf, ndmin=2)
                        df1 = ctf_local[index_in_film, 0]
                        df2 = ctf_local[index_in_film, 1]
                        angast = ctf_local[index_in_film, 2]

                    elif astigmatism:
                        df1 = df2 = ctf[0]  # TOMOCTFFIND
                        angast = 45.0
                    else:
                        df1 = ctf[2]  # CTFFIND3
                        df2 = ctf[3]  # CTFFIND3
                        angast = ctf[4]  # CTFFIND3
                    magnification = ctf[11]

                    # frealign_v9
                    occ = 100
                    sigma = 0.5
                    logp = change = 0
                    score = 0.5

                    f.write(
                        NEW_PAR_STRING_TEMPLATE
                        % (
                            count,
                            0.00,
                            0.00,
                            0.00,
                            0.00,
                            0.00,
                            float(magnification),
                            film,
                            df1,
                            df2,
                            angast,
                            occ,
                            logp,
                            sigma,
                            score,
                            change,
                        )
                    )
                    f.write("\n")

                    count += 1
                    index_in_film += 1
                film += 1
            # else:
            #    print 'No particles found for {}'.format(sname)
            #    inputlist.remove(sname)
        f.close()

    # TODO: check if mergeable with generateFrealignParFile
    @classmethod
    def generate_par_file(cls, input_par_file, output_par_file, mask):

        input = cls.from_file(input_par_file).data

        # generate new parameters
        count = 1
        for i in mask.split(","):
            sigma = float(i)
            if sigma > 0:
                input[:, count] += sigma * 2 * (np.random.rand(input.shape[0]) - 0.5)
            count += 1

        # write output file
        with open(output_par_file, "w") as f:

            # write header
            for line in open(input_par_file):
                if line.startswith("C"):
                    f.write(line)

            # write data
            for i in range(input.shape[0]):
                if input.shape[1] > 13:
                    f.write(NEW_PAR_STRING_TEMPLATE % tuple(input[i, :]))
                else:
                    f.write(CCLIN_PAR_STRING_TEMPLATE % tuple(input[i, :]))
                f.write("\n")

    @classmethod
    def from_file(cls, filename, toColumn=-1):
        """Reads parfile and creates Parameters object

        Parameters
        ----------
        filename : PathLike
            Parameter file

        Returns
        -------
        Parameters
            Object containing information in the input parfile
        """
        fieldwidths, fieldstring, version, extended = cls.format_from_parfile(filename)
        data, prologue, epilogue = cls.columns_from_parfile(
            filename, fieldwidths=fieldwidths, fieldstring=fieldstring, toColumn=toColumn
        )
        return Parameters(version, extended, data, prologue, epilogue)

    @classmethod
    def columns_from_parfile(cls, par_filename, fieldwidths=(), fieldstring="", toColumn=-1):
        """Obtains parfile data safely.

        Parameters
        ----------
        par_filename : PathLike
            Parameter file
        fieldwidths : tuple, optional
            Integer values specifying the text width of each field, by default ()
        fieldstring : str, optional
            String template for each parameter file entry, by default ""

        Returns
        -------
        data : numpy.ndarray
            Parameters in float
        prologue : list
            Rows of strings representing comments before main data
        epilogue : list
            Rows of strings representing comments after main data
        """

        import struct

        if not fieldwidths or not fieldstring:
            fieldwidths, fieldstring, _, _ = cls.format_from_parfile(par_filename)

        fmtstring = " ".join(
            "{}{}".format(abs(fw), "x" if fw < 0 else "s") for fw in fieldwidths
        )
        fieldstruct = struct.Struct(fmtstring)

        parse = fieldstruct.unpack_from

        # read array as string
        read_contents = False
        input = []
        prologue = []
        epilogue = []
        
        with open(par_filename) as f:
            lines = f.read().splitlines()
            
            # get header 
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith("C"):
                    prologue.append(line)
                else:
                    break
            
            # get footer
            for i in range(len(lines)-1, -1, -1):
                line = lines[i]
                if line.startswith("C"):
                    epilogue.append(line)
                else:
                    break
        
        if toColumn == -1:
            data_df = pd.read_csv(par_filename, 
                            comment='C', 
                            header=None, 
                            delim_whitespace=True)
        else:
             data_df = pd.read_csv(par_filename, 
                            comment='C', 
                            header=None, 
                            delim_whitespace=True,
                            usecols=[col for col in range(0,toColumn+1,1)])

        data = data_df.to_numpy()
        assert (not cls.has_nan(data)), "Your parfile has missing values NaN. Please check!"

        return data, prologue, epilogue

    @classmethod
    def format_from_parfile(cls, par_filename):
        """Figure out metric (parfile type) and return column sizes and format string.

        Parameters
        ----------
        par_filename : PathLike
            Parameter file

        Returns
        -------
        fieldwidths : tuple
            Integer values specifying the text width of each field
        fieldstring : str
            String template for each parameter file entry
        version : str
            One of (CCLIN, CC3M, NEW, FREALIGNX, CISTEM2)
        extended : bool
            Whether parfile is in extended format
        """
        columns = 0
        with open(par_filename) as f:
            for line in f:
                if (
                    not line.startswith("C")
                    and not line.startswith("_")
                    and not line.startswith("#")
                    and not line.startswith("data")
                    and not line.startswith("loop")
                    and not line == " \n"
                ):
                    columns = len(line.split())
                    break

        # logger.info("\nFound %d columns in %s" % (columns, par_filename))

        # frealignx
        if columns == FREALIGNX_NUM_COL:
            version = FREALIGNX
            extended = False
            fieldwidths = (7, 8, 8, 8, 10, 10, 8, 6, 9, 9, 8, 8, 8, 10, 11, 8, 8)
            fieldstring = FREALIGNX_PAR_STRING_TEMPLATE + "\n"
        # extended frealignx
        elif columns == EXTENDED_FREALIGNX_NUM_COL:
            version = FREALIGNX
            extended = True
            fieldwidths = (
                7,
                8,
                8,
                8,
                10,
                10,
                8,
                6,
                9,
                9,
                8,
                8,
                8,
                10,
                11,
                8,
                8,
                9,
                9,
                9,
                9,
                9,
                9,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            )
            fieldstring = EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE + "\n"
        # new
        elif columns == NEW_NUM_COL:
            version = NEW
            extended = False
            fieldwidths = (7, 8, 8, 8, 10, 10, 8, 6, 9, 9, 8, 8, 10, 11, 8, 8)
            fieldstring = NEW_PAR_STRING_TEMPLATE + "\n"  # extended cc3m
        elif columns == EXTENDED_NEW_NUM_COL:
            version = NEW
            extended = True
            fieldwidths = (
                7,
                8,
                8,
                8,
                10,
                10,
                8,
                6,
                9,
                9,
                8,
                8,
                10,
                11,
                8,
                8,
                9,
                9,
                9,
                9,
                9,
                9,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            )
            fieldstring = EXTENDED_NEW_PAR_STRING_TEMPLATE + "\n"
        # cclin
        elif columns == CCLIN_NUM_COL:
            version = CCLIN
            extended = False
            fieldwidths = (7, 8, 8, 8, 8, 8, 8, 6, 9, 9, 8, 7, 8)
            fieldstring = CCLIN_PAR_STRING_TEMPLATE + "\n"

        elif columns == EXTENDED_CCLIN_NUM_COL:
            version = CCLIN
            extended = False
            fieldwidths = (
                7,
                8,
                8,
                8,
                8,
                8,
                8,
                6,
                9,
                9,
                8,
                7,
                8,
                9,
                9,
                9,
                9,
                9,
                9,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
            )
            fieldstring = EXTENDED_CCLIN_PAR_STRING_TEMPLATE + "\n"
        else:
            logger.error(
                "Could not figure out file format. Columns = {0}".format(columns)
            )

        return fieldwidths, fieldstring, version, extended

    @classmethod
    def has_nan(cls, data: np.ndarray) -> bool:
        return np.isnan(np.sum(data))


    @staticmethod
    def get_particle_indexes(allparxs, particle):
        return [
            index
            for line, index in zip(allparxs, list(range(len(allparxs))))
            if float(line.split()[15]) == particle
        ]

    @staticmethod
    def remove_outliers(parfile, parameter, min, max):
        # shiftx, shifty, logp, sigma, score
        output_name = parfile.replace(".par", "_edit.par")
        if "score" in parameter:
            str_range = (120, 128)
            format = "%8.2f"
            str_length = 8
        elif "logp" in parameter:
            str_range = (99, 109)
            format = "%10.0f"
            str_length = 10
        elif "sigma" in parameter:
            str_range = (109, 120)
            format = "%11.4f"
            str_length = 11
        elif "shift" in parameter:
            if "x" in parameter:
                str_range = (31, 41)
                format = "%10.2f"
                str_length = 10
            elif "y" in parameter:
                str_range = (41, 51)
                format = "%10.2f"
                str_length = 10

        output_str = min

        with open(parfile, 'r') as f:
            with open(output_name, 'w') as out_file:
                for line in f.readlines():
                    if line.startswith('C'):
                        out_file.write(line)
                    else:
                        input_str = line[str_range[0]:str_range[1]]
                        if len(input_str.strip()) == str_length or float(input_str) < min or float(input_str) > max:
                            output_line = line[:str_range[0]] + format % output_str + line[str_range[1]:]
                        else:
                            output_line = line
                        out_file.write(output_line)
        os.remove(parfile)
        os.rename(output_name, parfile)

    @staticmethod
    def add_lines_with_statistics(input_parfile, current_class, current_path, is_frealignx=False):

        logger.debug("Adding 10,000 rows for good statistics with refine3d")
        # first append the other rows back
        film_col = 7
        current_data = Parameters.from_file(input_parfile).data
        film_id_now = current_data[-1, film_col]
        if is_frealignx:
            border = 17
            decoy = 16
        else:
            border = 16
            decoy = 15
        # hack_line = current_data[-1, :]
        # calculate sum and averages to add a hacking line
        frealign_part = current_data[:, 1:border]
        average1 = np.average(frealign_part, axis=0)
        var1 = np.var(frealign_part, axis=0)
        n1 = current_data.shape[0]
        n2 = 10000
        remote_par_stat = os.path.join(current_path, "frealign", "maps", "parfile_constrain_r%02d.txt" % current_class)
        par_stat = np.loadtxt(remote_par_stat, ndmin=2)
        averages = par_stat[0, :]
        vars = par_stat[1, :]
        # pshift if dose_weighting
        if is_frealignx:
            averages = np.insert(averages, 10, 0)
            vars = np.insert(vars, 10, 0)
        average2 = (averages * (n1 + n2) - (n1 * average1))/n2
        term1 = np.power((average1 - averages), 2)
        term2 = np.power((average2 - averages), 2)
        var2 = ((n1 + n2) * vars - n1 * var1 - n1 * term1 - n2 * term2)/n2
        var2 = np.abs(var2)
        # reset abnormal columns (film, df1, df2, angast)
        # var2[6:10] = 1
        std2 = np.sqrt(var2)
        hack_data = np.random.normal(size=(decoy, n2)) * std2[:, None] + average2[:, None]
        # hack_data = np.concatenate([np.random.normal(m, s, 100) for m, s in zip(average2, std2)])
        hack_data = np.transpose(hack_data)
        # renumber film id to add 1
        hack_data[:, 6] = film_id_now + 1
        # hard-code df1, df2
        hack_data[:, 7:9] = 20000 
        # hard-code angast
        hack_data[:, 9] = 0
        pid = np.arange((n1 + 1), (n1 + n2 + 1), 1, dtype=int).reshape(n2, 1)
        new_frealign_part = np.hstack((pid, hack_data))
        # add extended zeros
        extended = np.zeros((n2, 29), dtype=float)
        new_frealign_part = np.hstack((new_frealign_part, extended))
        column_num = current_data.shape[1]
        # overwrite the parx file for refine3d
        if column_num > 45:
            format = EXTENDED_FREALIGNX_PAR_STRING_TEMPLATE
        else:
            format = EXTENDED_NEW_PAR_STRING_TEMPLATE
            
        with open(input_parfile, 'a') as f:
            for i in range(new_frealign_part.shape[0]):
                f.write(format % tuple(new_frealign_part[i, :]) + "\n")
