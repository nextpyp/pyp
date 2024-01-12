



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




class Parameters:
    
    input_file: str = None
    data: np.ndarray = None
    num_columns: int = -1
    num_rows: int = -1

    def __init__(self, input_file: str):
        
        self.input_file = input_file
        
        if input_file.endswith(".cistem"):
            self.data = self.from_binary()
        elif input_file.endswith(".star"):
            self.data = self.from_star()

    
    @classmethod  
    def from_file(cls, input_file: str): 
        return Parameters(input_file=input_file)

    def from_binary(self, input_binary):
        return 
    
    def from_star(self, input_star):
        return 

    def to_binary(self):

    def to_star(self):

    def get_data(self):
        return self.data
