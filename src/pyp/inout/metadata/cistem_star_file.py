import numpy as np
import numpy.lib.recfunctions as rf

import sys
from pathlib import Path



"""
Constants goes here
"""
# data types (from cistem2/src/core/defines.h)
NONE = 0
TEXT = 1
INTEGER = 2
FLOAT = 3
BOOL = 4
LONG = 5
DOUBLE = 6
CHAR = 7
VARIABLE_LENGTH = 8
INTEGER_UNSIGNED = 9

# data parameter (from cistem2/src/core/cistem_parameters.h)
POSITION_IN_STACK = 1
IMAGE_IS_ACTIVE = 2
PSI = 4
X_SHIFT = 8
Y_SHIFT = 16
DEFOCUS_1 = 32
DEFOCUS_2 = 64
DEFOCUS_ANGLE = 128
PHASE_SHIFT = 256
OCCUPANCY = 512
LOGP = 1024
SIGMA = 2048
SCORE = 4096
SCORE_CHANGE = 8192
PIXEL_SIZE = 16384
MICROSCOPE_VOLTAGE = 32768
MICROSCOPE_CS = 65536
AMPLITUDE_CONTRAST = 131072
BEAM_TILT_X = 262144
BEAM_TILT_Y = 524288
IMAGE_SHIFT_X = 1048576
IMAGE_SHIFT_Y = 2097152
THETA = 4194304
PHI = 8388608
STACK_FILENAME = 16777216
ORIGINAL_IMAGE_FILENAME = 33554432
REFERENCE_3D_FILENAME = 67108864
BEST_2D_CLASS = 134217728
BEAM_TILT_GROUP = 268435456
PARTICLE_GROUP = 536870912
PRE_EXPOSURE = 1073741824
TOTAL_EXPOSURE = 2147483648
ASSIGNED_SUBSET = 4294967296
ORIGINAL_X_POSITION = 8589934592
ORIGINAL_Y_POSITION = 17179869184


# size of xxx in c++ in 64-bit arch
SIZE_INT = 4
SIZE_UINT = 4
SIZE_LONG = 8 # 4 in 32-bit
SIZE_FLOAT = 4
SIZE_CHAR = 1

# https://numpy.org/devdocs/reference/arrays.dtypes.html
# only used by from_binary()
MAPPING = {
    INTEGER: (SIZE_INT, 'i'),
    FLOAT: (SIZE_FLOAT, 'f'),
    LONG: (SIZE_LONG, 'i'),
    CHAR: (SIZE_CHAR, 'i'),
    INTEGER_UNSIGNED: (SIZE_UINT, 'u')
}

#####################
# Byte read pattern #
#####################
# 1. Rows and columns
DT_DIMS = np.dtype([('num_columns', '<i4'), ('num_rows', '<i4')])

# 2. Headers
DT_COLUMN = np.dtype([
    ('column_order', f'<i{SIZE_LONG}'),
    ('column_data', f'<i{SIZE_CHAR}'),
])

# Column name and corresponding size of data (in bytes) 
# from WriteTocisTEMBinaryFile of cistem2/src/core/cistem_parameters.cpp 
HEADER_LIST = {
    POSITION_IN_STACK: INTEGER_UNSIGNED,
    PSI: FLOAT,
    THETA: FLOAT, 
    PHI: FLOAT, 
    X_SHIFT: FLOAT, 
    Y_SHIFT: FLOAT, 
    DEFOCUS_1: FLOAT, 
    DEFOCUS_2: FLOAT, 
    DEFOCUS_ANGLE: FLOAT, 
    PHASE_SHIFT: FLOAT, 
    IMAGE_IS_ACTIVE: INTEGER, 
    OCCUPANCY: FLOAT, 
    LOGP: FLOAT, 
    SIGMA: FLOAT, 
    SCORE: FLOAT, 
    SCORE_CHANGE: FLOAT, 
    PIXEL_SIZE: FLOAT, 
    MICROSCOPE_VOLTAGE: FLOAT, 
    MICROSCOPE_CS: FLOAT, 
    AMPLITUDE_CONTRAST: FLOAT, 
    BEAM_TILT_X: FLOAT, 
    BEAM_TILT_Y: FLOAT, 
    IMAGE_SHIFT_X: FLOAT, 
    IMAGE_SHIFT_Y: FLOAT, 
    BEST_2D_CLASS: INTEGER, 
    BEAM_TILT_GROUP: INTEGER, 
    STACK_FILENAME: FLOAT, # wxString (not supported)
    ORIGINAL_IMAGE_FILENAME: FLOAT, # wxString (not supported)
    REFERENCE_3D_FILENAME: FLOAT, # wxString (not supported)
    PARTICLE_GROUP: INTEGER, 
    ASSIGNED_SUBSET: INTEGER, 
    PRE_EXPOSURE: FLOAT, 
    TOTAL_EXPOSURE: FLOAT, 
    ORIGINAL_X_POSITION: FLOAT, 
    ORIGINAL_Y_POSITION: FLOAT, 
}

def read_byte_str(f, size: int = 0):
    byte_str = f.read(size) if size > 0 else f.read()
    check_valid(byte_string=byte_str)
    return byte_str

def check_valid(byte_string: str):
    if not byte_string:
        raise Exception("Binary file is broken.") 
    

class Parameters:

    def __init__(self, input_file: str):
        
        self._input_file: str = None
        self._data: np.ndarray = None
        self._num_columns: int = -1
        self._num_rows: int = -1
        self._active_columns = []

        assert Path(input_file).exists(), f"{input_file} does not exist"

        if input_file.endswith(".cistem"):
            self.from_binary(input_binary=input_file)
        elif input_file.endswith(".star"):
            self.from_star(input_star=input_file)
        else:
            raise Exception(f"{input_file} file extension not recognized.")

        self._input_file = input_file

    @classmethod
    def from_file(cls, input_file: str): 
        return cls(input_file=input_file)

    def from_binary(self, input_binary):
        
        with open(input_binary, "rb") as f:

            # 1. read number of columns and rows            
            byte_str = read_byte_str(f, size=SIZE_INT+SIZE_INT)
            self._num_columns, self._num_rows = np.frombuffer(byte_str, dtype=DT_DIMS)[0]

            # 2. read headers to see which columns are active 
            for _ in range(self._num_columns):
                
                byte_str = read_byte_str(f, size=SIZE_LONG+SIZE_CHAR)
                byte_str_order, byte_str_data = np.frombuffer(byte_str, dtype=DT_COLUMN)[0]

                if byte_str_order in HEADER_LIST:
                    data_type = MAPPING[HEADER_LIST[byte_str_order]]
                    size_data_type, str_data_type = data_type[0], data_type[1]
                    self._active_columns.append((str(byte_str_order), f'<{str_data_type}{size_data_type}'))
                else:
                    raise Exception(f"Binary file contains unrecognized header. Column code = {byte_str_order}")
         
            assert len(self._active_columns) > 0, f"No column detected. Binary file might be broken."

            # 3. read the data and parse them into numpy array
            byte_str = read_byte_str(f)
            dt_data = np.dtype(self._active_columns)
            data = np.frombuffer(byte_str, 
                                 dtype=dt_data, 
                                 count=self.get_num_rows())
            data = np.array(data.tolist())

            # sanity check
            assert data.ndim == 2, "Data is not 2D" 
            assert (data.shape[0] == self.get_num_rows()), f"Number of rows does not match between data and header: {self.data.shape[0]} v.s. {self.num_rows}"
            assert (data.shape[1] == self.get_num_cols()), f"Number of columns does not match between data and header: {self.data.shape[1]} v.s. {self.num_columns}"

            self._data = data



    def from_star(self, input_star):
        raise Exception("Currently reading star file is not supported")
 
    def to_binary(self, filename):

        assert (self.get_input_file() is not None and self.get_data() is not None), "No data detected. Please read file first. "
        assert (self.get_num_cols() > 0 and self.get_num_rows() > 0), f"Number of columns ({self.num_columns}) and rows ({self.num_rows}) should be larger than 0"
        assert len(self._active_columns) > 0, f"No column detected."
        
        assert filename.endswith(".cistem"), f"Filename of output ({filename}) better has .cistem extension."

        with open(filename, "wb") as f:
            
            # 1. Write number of columns and rows
            byte_str = np.array([(self.get_num_cols(), self.get_num_rows())], dtype=DT_DIMS).tobytes()
            f.write(byte_str)

            # 2. Write headers
            for col in self._active_columns:
                str_data_order, str_data_type = col[0], col[1]
                bitmask_identifier = int(str_data_order)
                data_type = HEADER_LIST[bitmask_identifier]
                byte_str = np.array([(bitmask_identifier, data_type)], dtype=DT_COLUMN).tobytes()
                f.write(byte_str)
            
            # 3. Write data
            dt_data = np.dtype(self._active_columns)
            data_with_type = rf.unstructured_to_structured(self.get_data(), dtype=dt_data)
            f.write(data_with_type.tobytes())


    def to_star(self, filename):
        # TODO: simply call convert_binary_to_star executable
        return
    def get_data(self):
        return self._data
    def get_num_rows(self):
        return self._num_rows
    def get_num_cols(self):
        return self._num_columns
    def get_input_file(self): 
        # return the filename of input file
        return self._input_file
    
    def set_data(data: np.ndarray):
        # TODO: check if # columns is identical
        assert type(data) == np.ndarray, f"Input data type is not Numpy array, it is {type(data)}"
        assert data.ndim == 2, "Input data is not 2D"
        assert (self.get_data() is None or data.shape[1] == self.get_data().shape[1]), "Input data must have the same number of columns as original data."
        return 

test = Parameters.from_file("output.cistem")
test.to_binary("test.cistem")

test2 = Parameters.from_file("test.cistem")

