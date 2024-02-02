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

IMIND = 20
PIND = 15
TIND = 35
RIND = 70
FIND = 55

PSHIFT_X = 3
PSHIFT_Y = 9
PSHIFT_Z = 27
PPSI = 81
PTHETA = 273
PPHI = 819
ORIGINAL_X_POSITION_3D = 2457
ORIGINAL_Y_POSITION_3D = 7371
ORIGINAL_Z_POSITION_3D = 22113
PSCORE = 66339
POCC = 199017

TSHIFT_X = 7
TSHIFT_Y = 49
TILTANG = 343
TILTAXIS = 2401

FSHIFT_X = 11
FSHIFT_Y = 121


# byte size of xxx in c++ in 64-bit arch
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
DT_DIMS = np.dtype([('num_columns', f'<i{SIZE_INT}'), ('num_rows', f'<i{SIZE_INT}')])

# 2. Headers
DT_COLUMN = np.dtype([
    ('column_order', f'<i{SIZE_LONG}'),
    ('column_data', f'<i{SIZE_CHAR}'),
])

# 3. Block type 
DT_BLOCK = np.dtype([('block_type', f'<i{SIZE_LONG}')])

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
    IMIND: INTEGER, 
    PIND: INTEGER, 
    TIND: INTEGER, 
    RIND: INTEGER, 
    FIND: INTEGER, 
    PSHIFT_X: FLOAT, 
    PSHIFT_Y: FLOAT, 
    PSHIFT_Z: FLOAT, 
    PPSI: FLOAT, 
    PTHETA: FLOAT, 
    PPHI: FLOAT, 
    ORIGINAL_X_POSITION_3D: FLOAT, 
    ORIGINAL_Y_POSITION_3D: FLOAT, 
    ORIGINAL_Z_POSITION_3D: FLOAT, 
    PSCORE: FLOAT, 
    POCC: FLOAT, 
    TSHIFT_X: FLOAT, 
    TSHIFT_Y: FLOAT, 
    TILTANG: FLOAT, 
    TILTAXIS: FLOAT, 
    FSHIFT_X: FLOAT, 
    FSHIFT_Y: FLOAT, 
}

BLOCKS_EXTENDED = [PIND, TIND]

def read_byte_str(f, size: int = 0):
    byte_str = f.read(size) if size > 0 else f.read()
    check_valid(byte_string=byte_str)
    return byte_str

def check_valid(byte_string: str):
    if not byte_string:
        raise Exception("Binary file is broken.") 
    

class Parameters:

    HEADERS = [POSITION_IN_STACK, 
               PSI, 
               THETA, 
               PHI, 
               X_SHIFT, 
               Y_SHIFT, 
               DEFOCUS_1, 
               DEFOCUS_2, 
               DEFOCUS_ANGLE, 
               PHASE_SHIFT, 
               IMAGE_IS_ACTIVE, 
               OCCUPANCY, 
               LOGP, 
               SIGMA, 
               SCORE, 
               PIXEL_SIZE, 
               MICROSCOPE_VOLTAGE, 
               MICROSCOPE_CS, 
               AMPLITUDE_CONTRAST, 
               BEAM_TILT_X, 
               BEAM_TILT_Y, 
               IMAGE_SHIFT_X, 
               IMAGE_SHIFT_Y, 
               IMIND, 
               PIND, 
               TIND, 
               RIND, 
               FIND, 
               FSHIFT_X, 
               FSHIFT_Y
               ]

    def __init__(self, input_file: str = ""):
        
        self._input_file: Path = None
        self._data: np.ndarray = None
        self._num_columns: int = -1
        self._num_rows: int = -1
        self._active_columns = []

        if input_file.endswith(".cistem"):
            self.from_binary(input_binary=input_file)
        elif input_file.endswith(".star"):
            self.from_star(input_star=input_file)
        else:
            # create the class from scratch (given array and header)
            pass

        self._input_file = Path(input_file).absolute()

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
            assert (data.shape[0] == self.get_num_rows()), f"Number of rows does not match between data and header: {self.data.shape[0]} v.s. {self.get_num_rows()}"
            assert (data.shape[1] == self.get_num_cols()), f"Number of columns does not match between data and header: {self.data.shape[1]} v.s. {self.get_num_cols()}"

            self._data = data



    def from_star(self, input_star):
        raise Exception("Currently reading star file is not supported")
 
    def to_binary(self, filename):

        assert (self.get_num_cols() > 0 and self.get_num_rows() > 0), f"Number of columns ({self.get_num_cols()}) and rows ({self.get_num_rows()}) should be larger than 0"
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
    
    def get_index_of_column(self, column_code: int):
        return self.HEADERS.index(column_code)
    
    def set_data(self, data: np.ndarray):

        assert type(data) == np.ndarray, f"Input data type is not Numpy array, it is {type(data)}"
        assert data.ndim == 2, "Input data is not 2D"
        assert type(data[0]) == np.ndarray, f"First row of input data type is not Numpy array, it is {type(data[0])}"
        assert (self.get_data() is None or data.shape[1] == self.get_data().shape[1]), "Input data must have the same number of columns as original data."
        
        headers = self.HEADERS

        if len(headers) > 0:
            assert len(headers) == data.shape[1], f"Headers ({len(headers)}) do not have the same number of columns in the data ({data.shape[1]}). "
            # check if all the columns exist 
            for column_code in headers:
                assert (column_code in HEADER_LIST), f"{column_code} not in the Header List. "

            # reset the columns 
            self._active_columns.clear()

            for byte_str_order in headers:
                data_type = MAPPING[HEADER_LIST[byte_str_order]]
                size_data_type, str_data_type = data_type[0], data_type[1]
                self._active_columns.append((str(byte_str_order), f'<{str_data_type}{size_data_type}'))

        self._data = data
        self._num_rows = data.shape[0]
        self._num_columns = data.shape[1]

class Particle:

    """ Class storing particle data (3D sub-volumes)
    """

    def __init__(self, particle_index, shift_x, shift_y, shift_z, psi, theta, phi, x_position_3d, y_position_3d, z_position_3d, score, occ):
        self.particle_index = particle_index
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        self.psi = psi
        self.theta = theta
        self.phi = phi
        self.x_position_3d = x_position_3d
        self.y_position_3d = y_position_3d
        self.z_position_3d = z_position_3d
        self.score = score 
        self.occ = occ

class Tilt:

    """ Class storing tilt data (tilt geometry)
    """
    def __init__(self, tilt_index, region_index, shift_x, shift_y, angle, axis):
        self.tilt_index = tilt_index
        self.region_index = region_index
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.angle = angle
        self.axis = axis





class ExtendedParameters():

    # Headers
    HEADERS_PARTICLES = [PIND, PSHIFT_X, PSHIFT_Y, PSHIFT_Z, PPSI, PTHETA, PPHI, ORIGINAL_X_POSITION_3D, ORIGINAL_Y_POSITION_3D, ORIGINAL_Z_POSITION_3D, PSCORE, POCC]
    HEADERS_TILTS = [TIND, RIND, TSHIFT_X, TSHIFT_Y, TILTANG, TILTAXIS]

    def __init__(self, input_file: str = ""):
        self._input_file: Path = None
        self._particles: dict = {}
        self._tilts: dict = {}
        self._num_columns_particles: int = -1
        self._num_rows_particles: int = -1
        self._num_columns_tilts: int = -1
        self._num_rows_tilts: int = -1
        self._active_columns_particles = []
        self._active_columns_tilts = []

        if input_file.endswith(".cistem"):
            self.from_binary(input_binary=input_file)
        elif input_file.endswith(".star"):
            self.from_star(input_star=input_file)
        else:
            # create the class from scratch (given array and header)
            pass

        self._input_file = Path(input_file).absolute()

    @classmethod
    def from_file(cls, input_file: str): 
        return cls(input_file=input_file)
    

    def from_binary(self, input_binary):
        
        with open(input_binary, "rb") as f:

            # 1. read the block type (either particles or tilts data)
            byte_str = read_byte_str(f, size=SIZE_LONG)
            block_type = np.frombuffer(byte_str, dtype=DT_BLOCK)[0][0]

            # 2. read number of columns and rows            
            byte_str = read_byte_str(f, size=SIZE_INT+SIZE_INT)
            num_columns, num_rows = np.frombuffer(byte_str, dtype=DT_DIMS)[0]
            
            # 3. read headers to see which columns are active 
            active_columns = []
            bytes_per_line = 0
            for _ in range(num_columns):
                
                byte_str = read_byte_str(f, size=SIZE_LONG+SIZE_CHAR)
                byte_str_order, byte_str_data = np.frombuffer(byte_str, dtype=DT_COLUMN)[0]

                if byte_str_order in HEADER_LIST:
                    data_type = MAPPING[HEADER_LIST[byte_str_order]]
                    size_data_type, str_data_type = data_type[0], data_type[1]
                    active_columns.append((str(byte_str_order), f'<{str_data_type}{size_data_type}'))
                    bytes_per_line += size_data_type
                else:
                    raise Exception(f"Binary file contains unrecognized header. Column code = {byte_str_order}")
         
            assert len(active_columns) > 0, f"No column detected. Binary file might be broken."


            # 4. read the data and parse them into numpy array
            bytes_to_read = num_rows * (bytes_per_line)
            byte_str = read_byte_str(f, size=bytes_to_read)
            dt_data = np.dtype(active_columns)
            data = np.frombuffer(byte_str, 
                                 dtype=dt_data, 
                                 count=num_rows)
            data = np.array(data.tolist())

            # sanity check
            assert data.ndim == 2, "Data is not 2D" 
            assert (data.shape[0] == num_rows), f"Number of rows does not match between data and header: {self.data.shape[0]} v.s. {num_rows}"
            assert (data.shape[1] == num_columns), f"Number of columns does not match between data and header: {self.data.shape[1]} v.s. {num_columns}"

            if block_type == PIND:
                self._num_columns_particles = num_columns
                self._num_rows_particles = num_rows
                self._active_columns_particles = active_columns
                self._particles = self.convert_particles_array_to_dict(data)

            elif block_type == TIND:
                self._num_columns_tilts = num_columns
                self._num_rows_tilts = num_rows
                self._active_columns_tilts = active_columns
                self._tilts = self.convert_tilts_array_to_dict(data)
    
    
    def to_binary(self, filename):

        assert filename.endswith(".cistem"), f"Filename of output ({filename}) better has .cistem extension."

        with open(filename, "wb") as f:

            for block_identifier in BLOCKS_EXTENDED:
                    
                # 1. Write block identifier
                byte_str = np.array([(block_identifier)], dtype=DT_BLOCK).tobytes()
                f.write(byte_str)

                if block_identifier == PIND:
                    num_columns = self._num_columns_particles
                    num_rows = self._num_rows_particles
                    active_columns = self._active_columns_particles 
                    data = self.convert_particles_dict_to_array(self._particles)
                    
                elif block_identifier == TIND:
                    num_columns = self._num_columns_tilts
                    num_rows = self._num_rows_tilts
                    active_columns = self._active_columns_tilts
                    data = self.convert_tilts_dict_to_array(self._tilts)

                # 2. Write number of columns and rows
                byte_str = np.array([(num_columns, num_rows)], dtype=DT_DIMS).tobytes()
                f.write(byte_str)

                # 3. Write headers
                for col in active_columns:
                    str_data_order, str_data_type = col[0], col[1]
                    bitmask_identifier = int(str_data_order)
                    data_type = HEADER_LIST[bitmask_identifier]
                    byte_str = np.array([(bitmask_identifier, data_type)], dtype=DT_COLUMN).tobytes()
                    f.write(byte_str)
                
                # 4. Write data
                dt_data = np.dtype(active_columns)
                data_with_type = rf.unstructured_to_structured(data, dtype=dt_data)
                f.write(data_with_type.tobytes())
    
    def get_by_pind(self, pind: int) -> Particle:
        """get_by_pind Get particle parameters by particle index

        Parameters
        ----------
        pind : int
            Particle index

        Returns
        -------
        Particle
            Object containing particle information
        """
        if pind not in self._particles:
            return None
        return self._particles[pind]
    
    def get_by_tind_rind(self, tind: int, rind: int = 0) -> Tilt:
        """get_by_tind_rind Get tilt parameters by tilt index and region index

        Parameters
        ----------
        tind : int
            Tilt index
        rind : int
            Region index

        Returns
        -------
        Tilt
            Object containing tilt parameters
        """
        if tind not in self._tilts or rind not in self._tilts[tind]:
            return None
        return self._tilts[tind][rind]
    
    def get_input_file(self): 
        # return the filename of input file
        return self._input_file

    def convert_particles_array_to_dict(self, particles_data: np.ndarray) -> dict:

        assert type(particles_data) == np.ndarray, f"Tilt data type is not Numpy array, it is {type(particles_data)}"

        particles = {}
        
        def build_particle(row):
            particle_index = row[0]
            particle = Particle(particle_index=particle_index, 
                                shift_x=row[1], 
                                shift_y=row[2], 
                                shift_z=row[3], 
                                psi=row[4], 
                                theta=row[5], 
                                phi=row[6], 
                                x_position_3d=row[7], 
                                y_position_3d=row[8], 
                                z_position_3d=row[9], 
                                score=row[10], 
                                occ=row[11])
            
            assert particle_index not in particles, f"Particle index ({int(particle_index)}) appears twice in particle metadata"
            particles[particle_index] = particle

        [build_particle(row) for row in particles_data]

        return particles 
    
    def convert_particles_dict_to_array(self, particles: dict) -> np.ndarray:
        
        assert type(particles), f"Input is not a dictionary. It is {type(particles)}"
        assert len(particles.keys()) == self._num_rows_particles, f"Dictionary does not have the same number of particle. {len(particles.keys())} v.s. {self._num_rows_particles}"

        num_rows = self._num_rows_particles
        num_cols = self._num_columns_particles

        particles_arr = np.zeros((num_rows, num_cols))

        for line_counter, particle_index in enumerate(particles):
            particle = particles[particle_index]
            particles_arr[line_counter, :] = np.array([particle.particle_index,
                                                       particle.shift_x,
                                                       particle.shift_y,
                                                       particle.shift_z,
                                                       particle.psi,
                                                       particle.theta,
                                                       particle.phi,
                                                       particle.x_position_3d,
                                                       particle.y_position_3d,
                                                       particle.z_position_3d,
                                                       particle.score,
                                                       particle.occ])
                                                       


        return particles_arr
    
    def convert_tilts_array_to_dict(self, tilts_data: np.ndarray) -> dict:

        assert type(tilts_data) == np.ndarray, f"Tilt data type is not Numpy array, it is {type(tilts_data)}"

        tilts = {}

        def build_tilt(row):
            tilt_index = row[0]
            region_index = row[1]
            tilt = Tilt(tilt_index=tilt_index, 
                        region_index=region_index, 
                        shift_x=row[2], 
                        shift_y=row[3], 
                        angle=row[4],
                        axis=row[5])
            
            if tilt_index not in tilts:
                tilts[tilt_index] = {}

            assert region_index not in tilts[tilt_index], f"Region index ({int(region_index)}) appears twice in tilt metadata"            
            tilts[tilt_index][region_index] = tilt

        [build_tilt(row) for row in tilts_data]
        
        return tilts
    
    def convert_tilts_dict_to_array(self, tilts: dict) -> np.ndarray:

        assert type(tilts), f"Input is not a dictionary. It is {type(tilts)}"

        num_rows = self._num_rows_tilts
        num_cols = self._num_columns_tilts

        tilts_arr = np.zeros((num_rows, num_cols))

        line_counter = 0
        for tilt_index in tilts:
            for region_index in tilts[tilt_index]:

                tilt = tilts[tilt_index][region_index]
                tilts_arr[line_counter, :] = np.array([tilt.tilt_index,
                                                       tilt.region_index, 
                                                       tilt.shift_x,
                                                       tilt.shift_y,
                                                       tilt.angle,
                                                       tilt.axis])
                                                       
                line_counter += 1

        return tilts_arr

    def set_data(self, particles: dict, tilts: dict):

        assert type(particles), f"Input particles ({type(particles)}) needs to be a dictionary. "
        assert type(tilts), f"Input particles ({type(tilts)}) needs to be a dictionary. "

        # reset the columns 
        self._active_columns_particles.clear()
        self._active_columns_tilts.clear()

        for headers in [self.HEADERS_PARTICLES, self.HEADERS_TILTS]:
            active_columns = []
            for byte_str_order in headers:
                data_type = MAPPING[HEADER_LIST[byte_str_order]]
                size_data_type, str_data_type = data_type[0], data_type[1]
                active_columns.append((str(byte_str_order), f'<{str_data_type}{size_data_type}'))

            if headers[0] == PIND:
                self._active_columns_particles = active_columns.copy()
            elif headers[0] == TIND:
                self._active_columns_tilts = active_columns.copy()

        self._particles = particles
        self._tilts = tilts
        self._num_rows_particles = len(particles.keys())
        self._num_columns_particles = len(self.HEADERS_PARTICLES)
        self._num_rows_tilts = len([True for tilt_index in tilts for region_index in tilts[tilt_index]])
        self._num_columns_tilts = len(self.HEADERS_TILTS)
        



def initialize_parameters_binary(): 

    data = np.ones((100000000, 30))
    # data = np.random.rand(100, 30)

    new = Parameters()
    new.set_data(data=data)
    new.to_binary(filename="fresh.cistem")

    return 

def initialize_extended_parameters_binary():

    new = ExtendedParameters()    
    particles_arr = np.random.rand(10000, 12)
    particles_arr[:, 0] = np.array([i for i in range(10000)])
    tilts_arr = np.random.rand(10000, 6)
    particles_arr[:, 0] = np.array([i for i in range(10000)])

    particles = new.convert_particles_array_to_dict(particles_arr)
    tilts = new.convert_tilts_array_to_dict(tilts_arr)

    new.set_data(particles=particles, tilts=tilts)
    new.to_binary(filename="extended.cistem")

    return 


# test = Parameters.from_file("output.cistem")
# test.to_binary("test.cistem")

# test2 = Parameters.from_file("test.cistem")

# test_ext = ExtendedParameters.from_file("extended.cistem")
# particles = test_ext.get_particles_data()
# tilts = test_ext.get_tilts_data()


# initialize_parameters_binary()
# initialize_extended_parameters_binary()