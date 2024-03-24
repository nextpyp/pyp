
import os, sys
import glob
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from pyp.system.local_run import run_shell_command
from pyp.system.logging import initialize_pyp_logger
from pyp.system import project_params
from pyp.utils import get_relative_path, movie2regex, symlink_force
from pyp.system.utils import get_imod_path
from pyp.streampyp.logging import TQDMLogger
from pyp.inout.metadata import frealign_parfile
from pyp.analysis.geometry import getRelionMatrix, spk2Relion, relion2Spk, alignment2Relion, eulerZXZtoZYZ, eulerZYZtoZXZ
from pyp.analysis.geometry import transformations as vtk

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)


# lines to skip when reading text files
HEADERS = ["#", "C"]
# parfile columns 
PARHEADER = frealign_parfile.NEW_PAR_HEADER[-1].replace("\n", "").split()[1:]
PAREXTENDED = frealign_parfile.EXTENDED_NEW_PAR_HEADER[-1].replace("\n", "").split()[17:]
# PYP
FILES_PYP = {"pyp": (".pyp_config", "dict")}

# SPR schema
FILES_SPR = {"image": 
                {
                    "path": "%s.mrc", 
                    "format": "image",
                    "header": ["x", "y", "z"],
                    "index": None
                },
            "ctf": 
                {
                    "path": "%s.ctf", 
                    "format": "text", 
                    "header": ["ctf"],
                    "index": ["mean_df", "cc", "DF1", "DF2", "ANGAST", "ccc", 
                    "x", "y", "z", "pixel_size", "voltage", "magnification", "cccc", "counts" ],
                },
            "ctf_avrot": 
                {
                    "path": "%s_avgrot.txt", 
                    "format": "text", 
                    "header": ["col1", "col2", "col3", "col4", "col5", "col6"],
                    "index": None
                }, 
            "box": 
                {
                    "path": "%s.boxx", 
                    "format": "text", 
                    "header": ["x", "y", "Xsize", "Ysize", "inside", "selection"], 
                    "index": None
                },
            "drift": 
                {
                    "path": "%s.xf", 
                    "format": "text", 
                    "header": ["m00", "m01", "m02", "m03", "dx", "dy"], 
                    "index": None
                }
            }

# TOMO schema
FILES_TOMO= {"image": 
                {
                    "path": "%s.mrc", 
                    "format": "image",
                    "header": ["x", "y", "z"],
                    "index": None
                },
            "tomo": 
                {
                    "path": "%s.rec", 
                    "format": "image",
                    "header": ["x", "y", "z"],
                    "index": None
                },
            "order": 
                {
                    "path": "%s.order", 
                    "format": "text",
                    "header": ["order"],
                    "index": None
                },
            "ali": 
                {
                    "path": "%s.xf", 
                    "format": "text",
                    "header": ["m00", "m01", "m02", "m03", "dx", "dy"],
                    "index": None
                },
            "drift": 
                {
                    "path": "TILTSERIES_SCANORD_ANGLE.xf", 
                    "format": "text", 
                    "header": ["m00", "m01", "m02", "m03", "dx", "dy"],
                    "index": None
                }, 
            "frames": 
                {
                    "path": "frame_list.txt", 
                    "format": "text-str", 
                    "header": ["filename"],
                    "index": None
                },
            "tlt": 
                {
                    "path": "%s.tlt", 
                    "format": "text", 
                    "header": ["tilt_angle"], 
                    "index": None
                },
            "global_ctf": 
                {
                    "path": "%s.ctf", 
                    "format": "text", 
                    "header": ["_"],
                    "index": None
                }, 
            "ctf": 
                {
                    "path": "%s.def", 
                    "format": "text", 
                    "header": ["tilt_angle", "df1", "df2", "ast", "cc", "est_res"],
                    "index": None
                }, 
            "ctf_avrot": 
                {
                    "path": "%s_*_ctffind4_avrot.txt", 
                    "format": "text", 
                    "header": ["col1", "col2", "col3", "col4", "col5", "col6"],
                    "index": None
                }, 
            "ts_ctf_avgrot": 
                {
                    "path": "%s_avgrot.txt", 
                    "format": "text", 
                    "header": ["col1", "col2", "col3", "col4", "col5", "col6"],
                    "index": None
                }, 
            "ctf_tilt": 
                {
                    "path": "%s_????.txt", 
                    "format": "text", 
                    "header": ["counter", "df1", "df2", "angast", "cc", "res"],
                    "index": None
                }, 
            "box": 
                {
                    "path": "%s.spk", 
                    "format": "model", 
                    "header": ["x", "y", "z"], 
                    "index": None
                },
            "vir": 
                {
                    "path": "%s.vir", 
                    "format": "model", 
                    "header": ["x", "y", "z", "r"], 
                    "index": None
                },
            "exclude": 
                {
                    "path": "%s_exclude_views.mod", 
                    "format": "model", 
                    "header": ["x","y","exclude_view"],
                    "index": None
                }, 
            "gold3d": 
                {
                    "path": "%s_gold3d.mod", 
                    "format": "model", 
                    "header": ["x", "y", "z"], 
                    "index": None
                },
            "gold": 
                {
                    "path": "%s_gold.mod", 
                    "format": "model", 
                    "header": ["x", "y", "z"], 
                    "index": None
                },
            "web": 
                {
                    "path": "%s.pickle", 
                    "format": "pickle",
                    "header": ["web"],
                    "index": None
                },
            }


def getFilesByPattern(pattern, file_path, micrograph):
        
    root_pattern, file_format = os.path.splitext(pattern)

    regex = movie2regex(pattern, micrograph)
    r = re.compile(regex)
    
    directory = file_path.replace(os.path.basename(file_path), "")
    if len(directory) == 0: 
        directory = "./" 
    
    files = list(filter(r.match, os.listdir(directory)))

    try:
        pos_tiltangle = root_pattern.replace("TILTSERIES", "").split("_").index("ANGLE")
        files.sort(key=lambda x: float(x.replace(micrograph, "").replace(file_format, "").split("_")[pos_tiltangle]))
    except:
        pos_tiltangle = -1

    return list(map(lambda x: directory + x, files))  


class LocalMetadata:

    def __init__(self, filename, is_spr=True):

        # self.data
        # self.files
        # self.filename
        # self.micrograph
        # self.parameters 
        # self.counter
        
        
        self.filename = filename
        self.micrograph = os.path.splitext(os.path.basename(filename))[0] 
        self.parameters = None
        self.counter = 0

        if os.path.exists(filename):
            # try to open an existing pickle file
            # file = open(filename, "rb")
            try:
                # self.data = pickle.load(filename)
                self.data = pd.read_pickle(filename)
            except Exception as e:
                logger.error(f"Cannot read {filename}", e)
                self.data = {"mode": is_spr}
        else:
            # create a new one if it doesn't exist
            file = open(filename, "wb")
            self.data = {"mode": ""}

            assert ("mode" in self.data), "Input pickle file is invalid"

            # update the mode 
            if "tomo" not in self.data["mode"] and "spr" not in self.data["mode"]:
                self.data["mode"] = "spr" if is_spr else "tomo"
                pickle.dump(self.data, file)
        
            file.close()

        # determine metadata by mode
        self.files = FILES_SPR if "spr" in self.data["mode"] else FILES_TOMO
        
        # load pyp configuration if available
        try:
            self.loadPYPConfig()
        except:
            pass

    def __str__(self):
        # visulization of the metadata 
        output = ""
        for key in self.data.keys():
            output += f"\033[0;33m{key}: \033[0m"
            if type(self.data[key]) == str:
                output += self.data[key] 
            else:
                try:
                    row = self.data[key].shape[0]
                    col = self.data[key].shape[1]
                    output += f"({row} x {col})\n"
                    output += self.data[key].head(3).to_string() 
                except:
                    num = len(self.data[key])
                    output += f"({num} items)\n"

            output += "\n\n"
        return output


    def loadPYPConfig(self, file=".pyp_config.toml"):
        if os.path.exists(file):
            self.parameters = project_params.load_parameters()


    def isSPR(self):
        return True if "spr" in self.data["mode"] else False

    def isTomo(self):
        return True if "tomo" in self.data["mode"] else False

    def loadFiles(self, write=True):
        """ Load all the data based on data mode (spr or tomo)
        """
        for key in self.files:

            file_type = self.files[key]["format"]
            file_path = self.files[key]["path"]
            transpose = False
            files = []

            if "%s" in file_path:
                if key == "ctf_avrot":
                    # sort by indexes given by ctffind tilt ({micrograph}_{index}_ctffind4_avrot.txt)
                    files = glob.glob(file_path % (self.micrograph))
                    if self.isTomo():
                        files.sort(key=lambda x: int(x.replace(self.micrograph,"").split("_")[1]))
                    transpose = True
                if key == "ts_ctf_avgrot":
                    files = glob.glob(file_path % (self.micrograph))
                    transpose = True
                if key == "ctf_tilt":
                    # sort by indexes given by ctffind tilt ({micrograph}_{index}_ctffind4_avrot.txt)
                    files = glob.glob(file_path % (self.micrograph))
                    files.sort(key=lambda x: int(x.replace(self.micrograph,"").split("_")[1].replace(".txt","")))
                    transpose = True
                else:
                    files = glob.glob(file_path % (self.micrograph))
            elif key == "drift":

                if Path("frame_list.txt").exists():

                    files = open("frame_list.txt", "r").read().split("\n")

                    # it contains the name of movie frames instead of .xf
                    files = [str(Path(f).stem) + ".xf" for f in files]
                    files = [f for f in files if Path(f).exists()]

            elif key == "frames":
                if Path(file_path).exists():
                    files = [file_path]

            multiple_files = True if len(files) > 1 else False
            for f in files:
                if file_type == "text":
                    self.loadTextFile(key, f, multiple_files, transpose)
                elif file_type == "model":
                    self.loadModelFile(key, f)
                elif file_type == "par":
                    self.loadParFile(key, f)
                elif file_type == "image":
                    self.loadImageDim(key, f)
                elif file_type == "pickle":
                    self.loadPickleFile(key, f)
                elif file_type == "text-str":
                    self.loadTextStr(key, f)
                else:
                    raise Exception(f"File type {file_type} not recognized")

            self.counter = 0

        if write:
            self.write()

    def updateData(self, data):

        for key in data.keys():
            self.data[key] = data[key]


    def meta2PYP(self, path=".", data_path="."):
        """
        Write corresponding pyp files
        """
        name = self.micrograph
        if self.isSPR():
            for key in self.data.keys():
                if "drift" in key:
                    xf_motion = self.data["drift"].to_numpy()
                    np.savetxt(os.path.join(path, name + ".xf"), xf_motion, fmt="%s", delimiter='\t')
                elif key == "ctf":
                    ctf = self.data["ctf"].to_numpy()
                    np.savetxt(os.path.join(path, name + ".ctf"), ctf)
                elif key == "box":
                    boxx = self.data["box"].to_numpy()
                    if not os.path.exists(os.path.join(path, name + ".boxx")):
                        logger.info("Writing boxx files from metadata")
                        np.savetxt(os.path.join(path, name + ".boxx"), boxx, fmt="%s", delimiter='\t')
                    if not os.path.exists(os.path.join(path, name + ".box")):
                        np.savetxt(os.path.join(path, name + ".box"), boxx[:,:4], fmt="%s", delimiter='\t')
                elif key == "ctf_avrot":
                    avgrot = self.data["ctf_avrot"]
                    np.savetxt(os.path.join(path, name + "_avgrot.txt"), avgrot.T, fmt="%s", delimiter='\t')

        elif self.isTomo():
            prospective_keys = set(self.files.keys())
            existing_keys = prospective_keys.intersection(self.data.keys())
            for key in existing_keys:

                file_type = self.files[key]["format"]
                file_path = self.files[key]["path"]
                data = self.data[key]

                if file_type == "text":
                    transpose = False
                    template = None
                    if key == "drift" and self.parameters is not None:
                        if "frames" in self.data:
                            files = self.data["frames"]
                            template = [str(Path(f).stem) + ".xf" for f in files]
                    elif key == "ctf_avrot" and self.parameters is not None:
                        template = [str(self.files[key]["path"] % (self.micrograph)).replace("*", "%04d" % (index)) for index in data.keys()]
                        transpose = True
                    elif key == "ts_ctf_avgrot" and self.parameters is not None:
                        template = None
                        transpose = True
                    elif key == "ctf_tilt" and self.parameters is not None:
                        template = [str(self.files[key]["path"] % (self.micrograph)).replace("????", "%04d" % (index)) for index in data.keys()]
                        transpose = True
                    elif key == "global_ctf" and self.parameters is not None:
                        ctf = self.data["global_ctf"].to_numpy()
                        np.savetxt(os.path.join(path, name + ".ctf"), ctf)
                    try:
                        self.writeTextFile(data, file_path, output_path=path, template=template, transpose=transpose)
                    except:
                        pass
                elif file_type == "model":
                    self.writeModelFile(data, os.path.join(path, file_path))
                elif file_type == "tomo" or file_type == "image":
                    pass
                elif file_type == "pickle":
                    self.writePickleFile(data, os.path.join(path, file_path))
                elif file_type == "text-str":
                    self.writeTextStr(data, os.path.join(path, file_path))
                else:
                    raise Exception(f"File type {file_type} not recognized")
        else:
            raise Exception("Mode not recognized")  


    def loadTextFile(self, key, file, mutiple_files=False, transpose=False):
        """ Load data from text file 
        """
        assert (key in self.files), f"{key} not in the schema"
        assert (os.path.exists(file)), f"{file} does not exist"
        
        arr = np.loadtxt(file, ndmin=2, comments=HEADERS, dtype=float)

        if transpose:
            arr = arr.T
        header = self.files[key]["header"]
        index = self.files[key]["index"]

        assert (len(header) == arr.shape[1]), f"Headers do not match the dimension of array ({key},{file}): {len(header)} != {arr.shape[1]}"

        df = pd.DataFrame(arr, index=index, columns=header)

        if mutiple_files:
            if key not in self.data:
                self.data[key] = {}
            self.data[key][self.counter] = df
            self.counter += 1
        else:        
            self.data[key] = df

    def loadPickleFile(self, key, file):
        """ Load data from pickle file 
        """
        assert (key in self.files), f"{key} not in the schema"
        assert (os.path.exists(file)), f"{file} does not exist"
        
        with open(file,'rb') as f:
            self.data[key] = pickle.load(f)

    def loadModelFile(self, key, file):
        """ Load data from IMOD model file
        """
        assert (key in self.files), f"{key} not in the schema"
        assert (os.path.exists(file)), f"{file} does not exist"

        header = self.files[key]["header"]

        command = f"{get_imod_path()}/bin/imodinfo -a {file}"
        [output, error] = run_shell_command(command, verbose=False)

        modfile = output.split("contour")
        indexes = []
        if len(modfile) > 1:
            for c in range(1, len(modfile)):
                points_in_contour = int(modfile[c].split()[2])
                for point in range(points_in_contour):
                    indexes.append(
                        np.array(modfile[c].split("\n")[point + 1].split(), dtype=float)
                    )

        arr = np.array(indexes)
        # remove fourth column if using a 5-column model format
        if len(arr) > 0:
            if arr.shape[1] == 5:
                arr = np.delete( arr, 3, 1 )
            assert (len(header) == arr.shape[1]), f"Headers do not match the dimension of array ({key},{file}): {len(header)} != {arr.shape[1]}"
    
            df = pd.DataFrame(arr, columns=header)
            self.data[key] = df


    def loadImageDim(self, key, file):
        """ Load dimension of tomogram from .rec 
        """
        assert (key in self.files), f"{key} not in the schema"
        assert (os.path.exists(file)), f"{file} does not exist"

        header = self.files[key]["header"]

        command = f"{get_imod_path()}/bin/header -size '{file}'"
        [output, error] = run_shell_command(command, verbose=False)

        x, y, z = list(map(int, output.split()))
        if file.endswith(".rec"):
            arr = np.array([x, z, y])
        else:
            arr = np.array([x, y, z])
        arr.resize((1, 3))

        df = pd.DataFrame(arr, columns=header)
        self.data[key] = df

    def loadTextStr(self, key, file):
        assert (key in self.files), f"{key} not in the schema"
        assert (os.path.exists(file)), f"{file} does not exist"

        lst = [file for file in open(file, 'r').read().split("\n")]
        self.data[key] = lst

    def loadParFile(self, key, file):
        """ Load data from parfile
        """
        return 

    def update(self, key, filename, write=True):
        """ 
        """
        if write:
            self.write()

    def toArray(self, key):
        """ Return data in form of numpy array by key

        Args:
            key (str): key in the data dictionary
        """
        return 


    def write(self):
        """ Write out pickle file
        """
        if os.path.islink(self.filename):
            os.remove(self.filename)

        with open(self.filename, "wb") as file:
            pickle.dump(self.data, file)

    def writeTextFile(self, data, path, output_path=".", template=None, transpose=False):
        """ Produce text file(s) from numpy array

        Args:
            data (DataFrame): 2D data, it can also be numpy array
            path (str): output file name
            template (None or list, optional): List of file templates (it should be a list of raw frame movies). Defaults to None.
        """
        # if we have multiple files to output (i.e. frame alignment .xf in tomo)
        if template is not None:

            extension = os.path.splitext(os.path.basename(path))[1]
            self.counter = 0

            assert (type(data) == dict), "Data is expected to be a dictionary where keys are continuous indexes starting from 0"
            assert (len(data.keys()) == len(template)), f"Number of files do not match ({len(data.keys())} != {len(template)})"

            for t in template:
                template_ext = os.path.splitext(os.path.basename(t))[1]
                filename = os.path.join( output_path, os.path.basename(t).replace(template_ext, extension) )
                try:
                    d = data[self.counter]
                except KeyError:
                    raise Exception(f"{self.counter} is out of range 0-{len(data.keys())-1}.")

                if transpose:
                    np.savetxt(filename, d.T, fmt="%f\t")
                else:
                    np.savetxt(filename, d, fmt="%f\t")
                self.counter += 1

            self.counter = 0
        elif transpose:
            np.savetxt(path % (self.micrograph), data.T, fmt="%f\t")
        else:
            np.savetxt(path % (self.micrograph), data, fmt="%f\t")

    def writePickleFile(self, data, path):
        """ Produce pickle file(s)

        Args:
            data (DataFrame): 2D data, it can also be numpy array
            path (str): output file name
        """
        # if we have multiple files to output (i.e. frame alignment .xf in tomo)
        with open( path % (self.micrograph), 'wb') as f:
            pickle.dump(data, f)

    def writeTextStr(self, data, path):

        with open(path, "w") as f:
            f.write("\n".join(data))

    def writeModelFile(self, data, path):

        path = path % self.micrograph

        tmp = os.path.join( os.environ["PYP_SCRATCH"], "tmp.txt" )

        # swap y and z
        c = data.columns
        data[[c[1], c[2]]] = data[[c[2], c[1]]]

        np.savetxt(tmp, data.values, fmt="%f\t")

        # generate model from txt
        if data.shape[1] > 3:
            command = f"{get_imod_path()}/bin/point2model -scat -sphere 5 -values 1 {tmp} {path}"
        else:
            command = f"{get_imod_path()}/bin/point2model -scat -sphere 5 {tmp} {path}"
        run_shell_command(command, verbose=False)
        if os.path.exists(tmp):
            os.remove(tmp)

        # label yz-swap in the model file
        command = f"{get_imod_path()}/bin/imodtrans -Y -T {path} {path} && rm {path}~"
        run_shell_command(command, verbose=False)



    def getMissings(self):
        """ Get mssing entries in the metadata
        """
        return list(set(self.files.keys()) - set(self.data.keys()))


    def refresh_entries(self, parameters, update_virion=False):
        # clean metadata entries based on parameters from pyp
        meta_update = False
        # remove entries that need to be re-calculated
        if "ctf_force" in parameters and parameters["ctf_force"]:
            logger.info(
                f"CTF parameters will be re-computed"
            )
            if "ctf" in self.data:
                del self.data["ctf"]
                meta_update = True
            if "ctf_avrot" in self.data:
                del self.data["ctf_avrot"]
                meta_update = True
            if "ts_ctf_avgrot" in self.data:
                del self.data["ts_ctf_avgrot"]
                meta_update = True
            if "ctf_tilt" in self.data:
                del self.data["ctf_tilt"]
                meta_update = True

        if "movie_force" in parameters and parameters["movie_force"]:
            logger.info(
                f"Movie drift parameters will be re-computed"
            )
            if "drift" in self.data:
                del self.data["drift"]
                meta_update = True

        if "tomo_vir_force" in parameters and parameters["tomo_vir_force"]:
            if "vir" in self.data:
                logger.info(
                    f"Virion parameters will be re-computed"
                )
                del self.data["vir"]
                meta_update = True
            # also remove tomo spk
            if "box" in self.data:
                del self.data["box"]
                meta_update = True

        if "tomo_ali_force" in parameters and parameters["tomo_ali_force"]:
            logger.info(
                f"Tilt-series alignments will be re-computed"
            )
            if "ali" in self.data:
                del self.data["ali"]
                meta_update = True

        if "detect_force" in parameters and parameters["detect_force"] or update_virion:
            logger.info(
                f"Particle detection will be re-computed"
            )
            if "box" in self.data:
                del self.data["box"]
                meta_update = True

        # update current pkl file
        if meta_update:
            self.write()
            logger.info("Updating metadata (pkl) file")


def getTomoBinFactor(original_x, original_y, bin_tomo_x=512):

    squarex = math.ceil(original_x / (bin_tomo_x * 1.0)) * bin_tomo_x
    squarey = math.ceil(original_y / (bin_tomo_x * 1.0)) * bin_tomo_x
    square = max(squarex, squarey)

    return square, square / bin_tomo_x


def getTomoRefinement(tomo_name, matrix, tilt_angles, spike_counter, counter):

    BOX = 256
    center = BOX / 2
    low_tilt = tilt_angles[0][0]
    high_tilt = tilt_angles[-1][0]
    m = matrix
    spike_name = f"{tomo_name}_spk{spike_counter:04d}.rec"

    components = [counter, low_tilt, high_tilt, center, center, center, BOX, BOX, BOX, 0, 0, 0, 
                    m[0, 0], m[0, 1], m[0, 2], m[0, 3],
                    m[1, 0], m[1, 1], m[1, 2], m[1, 3],
                    m[2, 0], m[2, 1], m[2, 2], m[2, 3],
                    m[3, 0], m[3, 1], m[3, 2], m[3, 3], 
                    1, 1, 1, 0, spike_name]

    components = list(map(str, components))

    return "\t".join(components)


class GlobalMetadata:
    """
    Merge all the metadata and convert to different format
    """
    def __init__(self, name, parameters, imagelist="", mode="spr", getpickle=False, parfile="", path="."):
        self.name = name
        self.mode = mode
        self.data = {}
        self.refinement = pd.DataFrame()
        self.extended = pd.DataFrame()

        try: 
            self.scope_data = pd.DataFrame(
                {
                    "pixel_size": parameters["scope_pixel"], 
                    "AC": parameters["scope_wgh"], 
                    "fames": 60, 
                    "movie_Xsize": 5760, 
                    "movie_Ysize": 4092,
                    "voltage": parameters["scope_voltage"], 
                    "CS": parameters["scope_cs"], 
                    "total_dose": 60, 
                    "negative_stain": 0,
                    "phase_plate": 0,
                    "dose_rate": parameters["scope_dose_rate"],
                }, 
                index=["scope"]
                )

            if "tomo" in mode:
                self.tomo_rec = pd.DataFrame(
                    {
                        "tomo_rec_binning" : parameters["tomo_rec_binning"], 
                        "tomo_rec_thickness" : parameters["tomo_rec_thickness"], 
                        "tomo_rec_square" : "true" if parameters["tomo_rec_square"] else "false", 
                        "tomo_rec_format" : "true" if parameters["tomo_rec_format"] else "false",
                    }, 
                    index=["tomogram"]
                )

            self.micrograph_global = pd.DataFrame(
                {
                    "image_pixel_size": parameters["scope_pixel"], 
                    "image_Xsize": 5760, 
                    "image_Ysize": 4092, 
                    "ctf_hand": parameters["csp_ctf_handedness"] if "csp_ctf_handedness" in parameters else 1.0,
                }, 
                index=["micrograph"]
                )

            self.ptl_global = pd.DataFrame(
                {
                    "ptl_bin": int(parameters["extract_bin"] if "extract_bin" in parameters else 1),
                    "ptl_pixel_size": float(parameters["scope_pixel"]) * int(parameters["extract_bin"] if "extract_bin" in parameters else 1),  
                    "inversed": 0,
                    "normalized":  0,
                    "box_size": parameters["extract_box"] if "extract_box" in parameters else 0, 
                    "dimension": 2,
                }, 
                index=["particle"]
                )
        except: 
            logger.warning("Some PYP parameters are missing, metadata initialized with empty tables")
            self.scope_data = pd.DataFrame()
            self.micrograph_global = pd.DataFrame()
            self.ptl_global = pd.DataFrame()

        if getpickle and isinstance(imagelist, list):
            self.MergePickle(imagelist, path)

        if parfile:
            self.Read_GlobalParfile(parfile)
        else:
            logger.info(
                "Only global metadata will be exported"
                )

        # update image size if possible (from pkl files)
        if len(self.data.keys()) > 0:
            randome_micrograph = list(self.data.keys())[0]
            if "image" in self.data[randome_micrograph]:
                data = self.data[randome_micrograph]["image"].values[0]

                self.scope_data["movie_Xsize"] = data[0]
                self.scope_data["movie_Ysize"] = data[1]
                self.micrograph_global["image_Xsize"] = data[0]
                self.micrograph_global["image_Ysize"] = data[1]

    def MergePickle(self, imagelist, path="."):
        """
        Read individual local pickle files and merge the metadata ready to conversion
        """
        pickle_files = glob.glob(os.path.join(path, "*.pkl"))
        pickle_list = [x.split("/")[-1].replace(".pkl", "") for x in pickle_files] 
        
        assert not set(imagelist).difference(set(pickle_list)), f"Dataset images list not matching pickle files, check missing ones"
        merged_locals = {}
        for file in imagelist:
            picklefile = os.path.join(path, file + ".pkl")
            local_meta_data = pd.read_pickle(picklefile)
            merged_locals[file] = local_meta_data
        self.data.update(merged_locals)


    def WritePickle(self, path=".", to_pyp=False):
        
        imagelist = list(self.data.keys())
        if "spr" in self.mode:
            is_spr = True
        else:
            is_spr = False    
        for image in imagelist:
            localdata = self.data[image]
            localmeta = LocalMetadata(os.path.join(path, image + ".pkl"), is_spr = is_spr)
            localmeta.updateData(localdata)
            localmeta.write()
            if to_pyp:
                localmeta.meta2PYP(path=path)


    def UpdateGeneral(self, data=None, fromstar=True, frompar=False):
        if self.data and frompar:
            if "spr" in self.mode:
                self.micrograph_global.at[0, 
                    ["image_pixel_size", "image_Xsize", "image_Ysize" ]
                    ] = self.data[list(self.data.keys())[0]]["ctf"][["pixel_size", "x", "y"]].values
            elif "TOMO" in self.mode:
                pass
        elif not data.empty and fromstar:
            if "spr" in self.mode:
                self.scope_data.at[0, 
                    ["fames", "movie_Xsize", "movie_Ysize", "voltage", "dose_rate"]
                    ] = data[
                        [Relion.IMAGESIZEZ, Relion.IMAGESIZEX, Relion.IMAGESIZEY, Relion.VOLTAGE, Relion.MICROGRAPHDOSERATE]
                        ].values
                micrographsizeX = float(data[Relion.IMAGESIZEX].values[0]) * float(data[Relion.MICROGRAPHBIN].values[0])
                micrographsizeY = float(data[Relion.IMAGESIZEY].values[0]) * float(data[Relion.MICROGRAPHBIN].values[0])
                self.micrograph_global[["image_Xsize"]] = micrographsizeX
                self.micrograph_global[["image_Ysize"]] = micrographsizeY
 

    def UpdatePtlOptics(self, data, fromstar=True):
        
        if fromstar:
            if Relion.IMAGEPIXELSIZE in data.columns:
                self.ptl_global.at[0, "ptl_pixel_size"] = data[Relion.IMAGEPIXELSIZE].values[0]
            if Relion.IMAGESIZE in data.columns:
                self.ptl_global["box_size"] = data[Relion.IMAGESIZE].values[0]
            if Relion.IMAGEDIMENSION in data.columns:
                self.ptl_global["dimension"] = data[Relion.IMAGEDIMENSION].values[0]
            self.ptl_global["inversed"] = 1
            self.ptl_global["normalized"] = 1


    def UpdateDrift(self, imagename, xyarray, fromstar=True):
        if "spr" in self.mode and fromstar:
            if imagename not in self.data.keys():
                self.data.update({imagename: {}})
            drift_col = ["m00", "m01", "m02", "m03", "dx", "dy"]    
            frames = xyarray.shape[0]
            m00 = np.array([1] * frames).reshape(-1, 1)
            m01 = np.array([0] * frames).reshape(-1, 1)
            m02 = np.array([0] * frames).reshape(-1, 1)
            m03 = np.array([1] * frames).reshape(-1, 1)

            drift = np.hstack((m00, m01, m02, m03, xyarray))
            
            self.data[imagename]["drift"] = pd.DataFrame(drift, columns=drift_col)


    def UpdateCTF(self, imagename, data, fromstar=True):
        if "spr" in self.mode and fromstar:
            
            """
            {"ctf index": ["mean_df", "cc", "DF1", "DF2", "ANGAST", "ccc", 
                    "x", "y", "z", "pixel_size", "voltage", "magnification", "cccc", "counts" ]}
            CTF_PARAMS = [DEFOCUSU, DEFOCUSV, DEFOCUSANGLE, CS, PHASESHIFT, AC,
                BEAMTILTX, BEAMTILTY, BEAMTILTCLASS, CTFSCALEFACTOR, CTFBFACTOR,
                CTFMAXRESOLUTION, CTFFIGUREOFMERIT]
            """
            ctfindex = [
                "mean_df", "cc", "DF1", "DF2", "ANGAST", "ccc", 
                "x", "y", "z", "pixel_size", "voltage", "magnification", "cccc", "counts" 
            ]
            if imagename not in self.data.keys():
                self.data.update({imagename: {}})
            self.data[imagename]["ctf"] = pd.DataFrame(index=ctfindex, columns=["ctf"])
            # necessary values u, v, ast.values
            df1 = data[Relion.DEFOCUSU].values[0]
            df2 = data[Relion.DEFOCUSV].values[0] 
            angast = data[Relion.DEFOCUSANGLE].values[0]
            meandf = (float(df1) + float(df2)) / 2
            if Relion.CTFFIGUREOFMERIT in data.columns:          
                cc = data[Relion.CTFFIGUREOFMERIT].values[0]
                ccc = data[Relion.CTFFIGUREOFMERIT].values[0]
            else:
                cc = 0
                ccc = 0
            if Relion.CTFMAXRESOLUTION in data.columns:          
                cccc = data[Relion.CTFMAXRESOLUTION].values[0]
            else:
                cccc = 0 
            x = self.micrograph_global["image_Xsize"].values[0]
            y = self.micrograph_global["image_Ysize"].values[0]
            z = 1
            pixelsize = self.micrograph_global["image_pixel_size"].values[0]
            voltage = self.scope_data["voltage"].values[0]
            mag = 10000
            counts = 0

            self.data[imagename]["ctf"] = pd.DataFrame(np.array(
                [meandf, cc, df1, df2, angast, ccc, x, y, z, pixelsize, voltage, mag, cccc, counts], dtype=float
                ).reshape(-1,1), index=ctfindex, columns=["ctf"])


    def UpdateCoord(self, imagename, coordsarray, fromstar=True):
        
        if fromstar:
            if imagename not in self.data.keys():
                self.data.update({imagename: {}})

            boxsize = self.ptl_global["box_size"].values[0]
            ptl_psz = self.ptl_global["ptl_pixel_size"].values[0]
            original_pz = self.micrograph_global["image_pixel_size"].values[0]
            orginal_boxsize = boxsize * ptl_psz / original_pz
            ptl_num = coordsarray.shape[0]
            boxx = ["x", "y", "Xsize", "Ysize", "inside", "selection"]
            xsize = np.array([orginal_boxsize] * ptl_num).reshape(-1, 1)
            inside =  np.array([1] * ptl_num).reshape(-1, 1)
            selection =  np.array([0] * ptl_num).reshape(-1, 1)
            # center to the upleft corner
            coordsarray[:,0] = coordsarray[:,0] - xsize[0,0]/2
            coordsarray[:,1] = coordsarray[:,1] - xsize[0,0]/2
            box = np.hstack((coordsarray, xsize, xsize, inside, selection))
            self.data[imagename]["box"] = pd.DataFrame(box, columns=boxx)  


    def Read_GlobalParfile(self, parfile):
        """
        Read global refinement parfile to standard metadata
        """
        pardata = np.loadtxt(parfile, comments="C", ndmin=2)

        standard_par = pardata[:, :16]
        self.refinement = pd.DataFrame(standard_par, columns=PARHEADER)
        self.extended = pd.DataFrame(pardata[:, 16:], columns=PAREXTENDED)


    def meta2Star(self, filename, imagelist, select=1, stack="stack.mrcs", parfile="", frame_refinement=False, version="30001", output_path="."):
        """
        From metadata to star file for relion import
        """
        # refinement star
        version = "# version " + version + "\n"
        
        if "spr" in self.mode:

            # update micrograph names to film column
            coord = []
            newfilm = self.refinement["FILM"].copy()
            for id, imagename in enumerate(imagelist):
                
                relion_image_path = os.path.join(output_path, "Micrographs")
                relion_image = os.path.join(relion_image_path, imagename + ".mrc")
                
                mask = self.refinement["FILM"].astype(int).isin([id])
                newfilm.mask(mask, other=relion_image, inplace=True)
                
                # extract box coordinates, shift from left corner to center
                boxx = self.data[imagename]["box"].astype(int)
                boxx["x"] = boxx["x"] + boxx["Xsize"]/2
                boxx["y"] = boxx["y"] + boxx["Ysize"]/2
                coord.append(boxx.loc[(boxx["inside"] >= 1) & (boxx["selection"] >= select), "x" : "y"])

            self.refinement["FILM"] = newfilm

            self.refinement[["COORDX", "COORDY"]] = pd.concat(coord, axis=0, ignore_index=True)
            
            assert self.refinement["NO"].size == self.refinement["COORDX"].size, f"Particle number is not equal to box coordinates number"

            optics_header = """

data_optics

loop_ 
_rlnOpticsGroup #1 
_rlnOpticsGroupName #2 
_rlnAmplitudeContrast #3 
_rlnSphericalAberration #4 
_rlnVoltage #5 
_rlnImagePixelSize #6 
_rlnMicrographOriginalPixelSize #7 
_rlnImageSize #8 
_rlnImageDimensionality #9 
"""
            data_particles_header = """
data_particles

loop_ 
_rlnImageName #1 
_rlnMicrographName #2 
_rlnCoordinateX #3 
_rlnCoordinateY #4 
_rlnAnglePsi #5 
_rlnAngleTilt #6  
_rlnAngleRot #7 
_rlnDefocusU #8 
_rlnDefocusV #9 
_rlnDefocusAngle #10 
_rlnOriginXAngst #11 
_rlnOriginYAngst #12 
_rlnPhaseShift #13 
_rlnOpticsGroup #14
_rlnGroupNumber #15 
_rlnRandomSubset #16 
        """
            
            saved_file = os.path.join(output_path, filename)
            if not frame_refinement:
                ac = self.scope_data["AC"].values[0]
                cs = self.scope_data["CS"].values[0]
                voltage = self.scope_data["voltage"].values[0]
                ptl_pxl = self.ptl_global["ptl_pixel_size"].values[0]
                # get values and write refine star
                if not os.path.exists(stack):
                    optics_group = 1
                    optics_groupname = "opticsGroup" + str(optics_group)
                    image_original_pxl = self.micrograph_global["image_pixel_size"].values[0]
                    ptl_size = self.ptl_global["box_size"].values[0]
                    ptl_dimension = self.ptl_global["dimension"].values[0]

                    data_optics = version  +  optics_header 
                    data_optics_value = f"\n{optics_group}  {optics_groupname}  {ac}    {cs}    {voltage}   {ptl_pxl}   {image_original_pxl}    {ptl_size}  {ptl_dimension} \n\n"
                    data_optics_str = data_optics + data_optics_value

                    shifts = - (self.refinement[["SHX", "SHY"]].astype(int))

                    align_ctf = self.refinement[["PSI", "THETA", "PHI", "DF1", "DF2", "ANGAST"]]
                    micrograph_coord = self.refinement[["FILM", "COORDX", "COORDY"]]
                    total_ptl = int(self.refinement.iloc[-1, 0])
                    length = len(str(total_ptl))
                    ptl_name = pd.DataFrame([f"{i:0{length}d}@stack.mrcs" for i in range(1, total_ptl + 1)], columns=["PTL_NAME"])
                    phase_op_group = pd.DataFrame(
                        {
                            "PHASE": np.array([0] * total_ptl), 
                            "OPTGROUP": np.array([1] * total_ptl),
                            "GROUPNUM": np.array([1] * total_ptl), 
                        }
                    )

                    randomsubset = pd.DataFrame(np.random.randint(1, high=3, size=total_ptl, dtype=int), columns=["RAND_SUBSET"])
                    columns = [ptl_name, micrograph_coord, align_ctf, shifts, phase_op_group, randomsubset]
                    star_columns = pd.concat(columns, axis=1)
                    star_header = data_optics_str + version + data_particles_header
                    npvalue = star_columns.to_numpy(dtype=str, copy=True)

                    np.savetxt(saved_file, npvalue, fmt='%s', header=star_header, delimiter="\t", comments='')

                else:

                    comm = "par2star.py --stack {0} --apix {1} --ac {2} --cs {3} --voltage {4} {5} {6}".format(
                        stack, ptl_pxl, ac, cs, voltage, parfile, saved_file
                    )
                    run_shell_command(comm, verbose=True)
                logger.info(f"Alignments exported to {saved_file}")
            else:
                # mostly using stack instead of exporting raw shifts
                pass

        elif "tomo" in self.mode:
        # tomo conversion
            tomo_data_header = """
# version 30001

data_global

loop_ 
_rlnTomoName #1 
_rlnTomoTiltSeriesName #2 
_rlnTomoFrameCount #3 
_rlnTomoSizeX #4 
_rlnTomoSizeY #5 
_rlnTomoSizeZ #6 
_rlnTomoHand #7 
_rlnOpticsGroupName #8 
_rlnTomoTiltSeriesPixelSize #9 
_rlnVoltage #10 
_rlnSphericalAberration #11 
_rlnAmplitudeContrast #12 
_rlnTomoImportFractionalDose #13 
"""
        
        # tomo tilt would be one block for each tilt series
        # tomo tilt data: data_{tilt_name}
            tomo_tilt_header = """
# version 30001

data_%s

loop_ 
_rlnTomoProjX #1 
_rlnTomoProjY #2 
_rlnTomoProjZ #3 
_rlnTomoProjW #4 
_rlnDefocusU #5 
_rlnDefocusV #6 
_rlnDefocusAngle #7 
_rlnCtfScalefactor #8 
_rlnMicrographPreExposure #9 
"""
        # subtomograms refinement star
            optics_header = """
# version 30001

data_optics

loop_ 
_rlnOpticsGroup #1 
_rlnOpticsGroupName #2 
_rlnSphericalAberration #3 
_rlnVoltage #4 
_rlnTomoTiltSeriesPixelSize #5 
_rlnCtfDataAreCtfPremultiplied #6 
_rlnImageDimensionality #7 
_rlnTomoSubtomogramBinning #8 
_rlnImagePixelSize #9 
_rlnImageSize #10
        """
            data_particles_header = """
# version 30001

data_particles

loop_ 
_rlnTomoName #1 
_rlnTomoParticleId #2 
_rlnTomoManifoldIndex #3 
_rlnCoordinateX #4 
_rlnCoordinateY #5 
_rlnCoordinateZ #6 
_rlnOriginXAngst #7 
_rlnOriginYAngst #8 
_rlnOriginZAngst #9 
_rlnAngleRot #10 
_rlnAngleTilt #11 
_rlnAnglePsi #12 
_rlnClassNumber #13 
_rlnRandomSubset #14 
_rlnTomoParticleName #15 
_rlnOpticsGroup #16 
_rlnImageName #17 
_rlnCtfImage #18 
_rlnGroupNumber #19 
_rlnNormCorrection #20 
_rlnLogLikeliContribution #21 
_rlnMaxValueProbDistribution #22 
_rlnNrOfSignificantSamples #23
        """
            particles_header = """
# version 30001 by xpytools

data_particles

loop_
_rlnTomoName #1
_rlnTomoParticleId #2
_rlnTomoManifoldIndex #3
_rlnCoordinateX #4
_rlnCoordinateY #5
_rlnCoordinateZ #6
_rlnOriginXAngst #7
_rlnOriginYAngst #8
_rlnOriginZAngst #9
_rlnAngleRot #10
_rlnAngleTilt #11
_rlnAnglePsi #12
_rlnClassNumber #13
_rlnRandomSubset #14
_rlnLogLikeliContribution #15
"""

        # frame refined motion star, one block for each subtomogram
            motion_header = """
# version 30001

data_general

_rlnParticleNumber                   13320


# version 30001

data_TS_01/1

loop_
_rlnOriginXAngst #1
_rlnOriginYAngst #2
_rlnOriginZAngst #3
        """

            # global parameters
            pixel_size = self.scope_data["pixel_size"].values[0]
            voltage = self.scope_data["voltage"].values[0]
            cs = self.scope_data["CS"].values[0]
            ac = self.scope_data["AC"].values[0]
            dose_rate = self.scope_data["dose_rate"].values[0]

            dataset, format = os.path.splitext(filename)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            tomogram_file = os.path.join( os.path.abspath(output_path), f"relion/{dataset}_tomograms{format}")
            particle_file = os.path.join( os.path.abspath(output_path), f"relion/{dataset}_particles{format}")

            EXTEND_START = 16

            FILM_COL = 8 - 1
            SCANORD_COL = 20 - 1
            PTLIND_COL = 17 - 1

            TILTAN_COL = 18 - 1
            NOMRX_COL = 24 - 1
            NORMZ_COL = 26 - 1
            MATRIX0_COL = 27 - 1
            MATRIX12_COL = 39 - 1
            MATRIX13_COL = 40 - 1
            MATRIX15_COL = 42 - 1
            PPSI_COL = 43 - 1
            PPHI_COL = 45 - 1

            # 2 star files are required:
            # tomogram.star (for ImportTomo), coord.star (for ImportParticle)

            #################
            # tomogram.star #
            #################

            header = tomo_data_header
            body = ""

            logger.info(f"Exporting metadata for {len(self.data)} tomograms")
            with tqdm(desc="Progress", total=len(self.data), file=TQDMLogger()) as pbar:
                for micrograph in self.data.keys():
                    film_index = imagelist.index(micrograph)
                    data = self.data[micrograph]

                    # link tilt-series .mrc to relion folder if needed
                    if not os.path.exists(os.path.join(output_path, "relion", "Movies")):
                        os.makedirs(os.path.join(output_path, "relion", "Movies"))
                    if not os.path.exists(os.path.join(output_path, "relion", "Movies", f"{micrograph}.mrc")):
                        os.symlink(os.path.join(os.getcwd(), "mrc", f"{micrograph}.mrc"), os.path.join(output_path, "relion", "Movies", f"{micrograph}.mrc"))

                    # raw image size
                    num_tilts = data["image"].values[0][-1]
                    x = data["image"].values[0][0]
                    y = data["image"].values[0][1]

                    tomo_x = data["tomo"].values[0][0]
                    tomo_y = data["tomo"].values[0][1]
                    # tomo_z = data["tomo"].values[0][2]

                    # square, binning = getTomoBinFactor(x, y, bin_tomo_x=tomo_x)
                    binning = self.tomo_rec.loc["tomogram", "tomo_rec_binning"]
                    full_tomo_x = tomo_x * binning
                    full_tomo_y = tomo_y * binning
                    z = self.tomo_rec.loc["tomogram", "tomo_rec_thickness"]

                    # not sure what they are
                    hand = -1.0 if not self.micrograph_global["ctf_hand"].values[0] else 1.0
                    optic_group_name = "opticsGroup1"

                    micrograph_optics = list(map(str, [micrograph, f"Movies/{micrograph}.mrc", num_tilts, full_tomo_x, full_tomo_y, z, hand, optic_group_name, pixel_size, voltage, cs, ac, dose_rate]))
                    header += "\t".join(micrograph_optics)
                    header += "\n"

                    body += "\n\n"
                    body += tomo_tilt_header % (micrograph)
                    for tilt in range(num_tilts):

                        tilt_angle = data["tlt"].values[tilt][0]
                        xf = data["ali"].values[tilt]

                        df1 = data["ctf"].values[tilt][1]
                        df2 = data["ctf"].values[tilt][2]
                        astang = data["ctf"].values[tilt][3]
                        ctf_scale = 1.0
                        scanord = data["order"].values[tilt][0]
                        exposure = scanord * dose_rate

                        # add csp tilt parameters to xf
                        dx_tilt = dy_tilt = 0.0
                        condition = np.where(
                                        (self.refinement.values[:, FILM_COL] == film_index) & \
                                        (self.extended.values[:, SCANORD_COL - EXTEND_START] == scanord)
                                    )
                        if condition[0].size != 0:
                            tilt_data = self.extended.values[condition, :][0, 0, :]
                            dx_tilt, dy_tilt = tilt_data[MATRIX12_COL - EXTEND_START : MATRIX13_COL - EXTEND_START + 1]
                            tilt_angle = tilt_data[TILTAN_COL - EXTEND_START]

                        xf[4] -= dx_tilt / pixel_size
                        xf[5] -= dy_tilt / pixel_size

                        matrix = getRelionMatrix(tilt_angle, xf, z, [x, y], full_tomo_x, full_tomo_y)
                        for r in range(matrix.shape[0]):
                            body += f"[{matrix[r,0]:.10f},{matrix[r,1]:.10f},{matrix[r,2]:.10f},{matrix[r,3]:.10f}] "


                        micrograph_tilt = list(map(str, [df1, df2, astang, ctf_scale, exposure]))
                        body += "\t".join(micrograph_tilt) + "\n"

                    pbar.update(1)

            header += body
            with open(tomogram_file, "w") as f:
                f.write(header)
            logger.info(f"Tomogram metadata exported to {tomogram_file}")

            ###### end of tomogram.star #######

            #################
            #   coord.star  #
            #################
            header = particles_header

            counter = 1
            manifold = 1
            class_num = 1
            random_subset = 2

            logger.info(f"Exporting particle metadata from {len(self.data)} tomograms")
            with tqdm(desc="Progress", total=len(self.data), file=TQDMLogger()) as pbar:
                for micrograph in self.data.keys():
                    film_index = imagelist.index(micrograph)
                    data = self.data[micrograph]
                    x = data["image"].values[0][0]
                    y = data["image"].values[0][1]
                    tomo_x = data["tomo"].values[0][0]
                    tomo_y = data["tomo"].values[0][1]
                    tomo_z = data["tomo"].values[0][2]

                    binning = self.tomo_rec.loc["tomogram", "tomo_rec_binning"]
                    full_tomo_x = tomo_x * binning
                    full_tomo_y = tomo_y * binning
                    full_thickness = self.tomo_rec.loc["tomogram", "tomo_rec_thickness"]

                    coordinates = data["box"].values
                    for particle_index, coord in enumerate(coordinates):
                        x, y, z = coord
                        relion_x, relion_y, relion_z = spk2Relion(x, y, z, binning, full_tomo_x, full_tomo_y, thickness=full_thickness, tomo_x_bin=tomo_x, tomo_y_bin=tomo_y, tomo_z_bin=tomo_z)

                        # try different scanning orders to get particle alignment
                        # NOTE: ensure ptlind should match the index in spk file
                        for scanord in range(0, 40, 5):
                            condition = np.where(
                                            (self.refinement.values[:, FILM_COL] == film_index) & \
                                            (self.extended.values[:, PTLIND_COL - EXTEND_START] == particle_index) & \
                                            (self.extended.values[:, SCANORD_COL - EXTEND_START] == scanord)
                                        )
                            if condition[0].size != 0:
                                break

                        # if particle is not in the parfile
                        if condition[0].size == 0:
                            continue

                        particle_data = self.extended.values[condition, :][0, 0, :]
                        ptl_condition = np.where(
                                            (self.refinement.values[:, FILM_COL] == film_index) & \
                                            (self.extended.values[:, PTLIND_COL - EXTEND_START] == particle_index) & \
                                            (np.abs(self.extended.values[:, TILTAN_COL - EXTEND_START]) <= 20)
                                        )

                        particle_score = np.mean(self.refinement.values[ptl_condition, :][:, :, 14])
                        if np.isnan(particle_score):
                            particle_score = 0
                        matrix = particle_data[MATRIX0_COL - EXTEND_START : MATRIX15_COL - EXTEND_START + 1]
                        ppsi, ptheta, pphi = particle_data[PPSI_COL - EXTEND_START : PPHI_COL - EXTEND_START + 1]
                        normX, normY, normZ = particle_data[NOMRX_COL - EXTEND_START : NORMZ_COL - EXTEND_START + 1]
                        rot, tilt, psi, dx, dy, dz = alignment2Relion(matrix, ppsi, ptheta, pphi, normX, normY, normZ)

                        # relion will reset translation to zero during importing particles
                        # so we add the translation to coordinates
                        relion_x -= dx / pixel_size
                        relion_y -= dy / pixel_size
                        relion_z -= dz / pixel_size
                        dx = dy = dz = 0.0

                        particle = list(map(str, [micrograph, counter, manifold, relion_x, relion_y, relion_z, f"{dx:.3f}", f"{dy:.3f}", f"{dz:.3f}", f"{rot:.2f}", f"{tilt:.2f}", f"{psi:.2f}", class_num, random_subset, particle_score]))
                        header += "\t".join(particle) + "\n"

                        counter += 1
                        class_num *= -1
                        random_subset = 1 if random_subset == 2 else 2
                    pbar.update(1)

            with open(particle_file, "w") as f:
                f.write(header)
            logger.info(f"Particle metadata exported to {particle_file}")

            ###### end of coord.star #######


    def weak_meta2Star(self, imagelist, filename, input_dir, coords=True, version="30001"):
        """
        From metadata to star file for relion import
        """
        # refinement star
        version = "# version " + version + "\n"
        current_dir = os.getcwd()

        if "spr" in self.mode:

            # update micrograph names to film column
            image_name = []
            coord = []
            ctf = []
            # newfilm = self.refinement["FILM"].copy()
            relion_image_path = "Micrographs/"

            Path("relion/Micrographs").mkdir(parents=True, exist_ok=True)

            # relion_films = [relion_image_path + x + ".mrc" for x in imagelist]

            for id, imagename in enumerate(imagelist):

                # link averages to Micrographs
                target = f"{current_dir}/relion/Micrographs/{imagename}.mrc"
                if os.path.exists(target):
                    try:
                        os.remove(target)
                    except OSError:
                        pass
                try:
                    os.symlink(os.path.join(input_dir,"mrc",imagename + ".mrc"), target)
                except OSError:
                    pass

                ctfmeta = self.data[imagename]["ctf"]
                ctf_row = ctfmeta.T
                ctf_mic = ctf_row[["DF1", "DF2", "ANGAST", "ccc", "cccc"]]
                # extract box coordinates, shift from left corner to center
                if coords:
                    if "box" in self.data[imagename].keys():
                        boxx = self.data[imagename]["box"].astype(int)
                        boxx["x"] = boxx["x"] + boxx["Xsize"]/2
                        boxx["y"] = boxx["y"] + boxx["Ysize"]/2
                        coord.append(boxx.loc[(boxx["inside"] >= 1) & (boxx["selection"] >= 0), "x" : "y"])

                        # repeat the ctf information to match particles rows
                        repeat = len(coord[-1])
                        ctf_per_image = pd.concat([ctf_mic] * repeat, ignore_index=True)

                        # repeat image names
                        film = [relion_image_path + imagename + ".mrc"] * repeat
                        relion_films = pd.DataFrame({"FILM": film})
                    else:
                        continue
                else:
                    # coord = pd.DataFrame({"COORDX":[0], "COORDY":[0]})
                    ctf_per_image = ctf_mic
                    relion_films = pd.DataFrame({"FILM": [relion_image_path + imagename + ".mrc"]})

                ctf.append(ctf_per_image)
                image_name.append(relion_films)

            self.refinement["FILM"] = pd.concat(image_name, axis=0, ignore_index=True)
            self.refinement[["DF1", "DF2", "ANGAST", "CTF_MERIT", "CTF_MAX_RESOLUTION"]] = pd.concat(ctf, axis=0, ignore_index=True)

            if coords:
                self.refinement[["COORDX", "COORDY"]] = pd.concat(coord, axis=0, ignore_index=True)

            optics_header = """

data_optics

loop_ 
_rlnOpticsGroup #1 
_rlnOpticsGroupName #2 
_rlnAmplitudeContrast #3 
_rlnSphericalAberration #4 
_rlnVoltage #5 
_rlnImagePixelSize #6 
_rlnMicrographOriginalPixelSize #7 

"""
            if coords:
                data_particles_header = """
data_particles

loop_ 
_rlnImageName #1 
_rlnMicrographName #2 
_rlnCoordinateX #3 
_rlnCoordinateY #4 
_rlnDefocusU #5
_rlnDefocusV #6 
_rlnDefocusAngle #7 
_rlnCtfFigureOfMerit #8 
_rlnCtfMaxResolution $9
_rlnPhaseShift #10
_rlnOpticsGroup #11
_rlnGroupNumber #12 
_rlnRandomSubset #13 
"""
            else:
                data_particles_header = """
data_particles

loop_ 
_rlnMicrographName #1
_rlnDefocusU #2
_rlnDefocusV #3 
_rlnDefocusAngle #4 
_rlnCtfFigureOfMerit #5 
_rlnCtfMaxResolution $6
"""

            ac = self.scope_data["AC"].values[0]
            cs = self.scope_data["CS"].values[0]
            voltage = self.scope_data["voltage"].values[0]
            ptl_pxl = self.ptl_global["ptl_pixel_size"].values[0]

            # get values and write refine star
            optics_group = 1
            optics_groupname = "opticsGroup" + str(optics_group)
            image_original_pxl = self.micrograph_global["image_pixel_size"].values[0]

            data_optics = version  +  optics_header 
            data_optics_value = f"\n{optics_group}  {optics_groupname}  {ac}    {cs}    {voltage}   {ptl_pxl}   {image_original_pxl} \n\n"
            data_optics_str = data_optics + data_optics_value

            # shifts = - (self.refinement[["SHX", "SHY"]].astype(int))

            CTFs = self.refinement[["DF1", "DF2", "ANGAST", "CTF_MERIT", "CTF_MAX_RESOLUTION"]]
            if coords: 
                micrograph_coord = self.refinement[["FILM", "COORDX", "COORDY"]]
                total_ptl = micrograph_coord.shape[0]
                length = len(str(total_ptl))
                ptl_name = pd.DataFrame([f"{i:0{length}d}@stack.mrcs" for i in range(1, total_ptl + 1)], columns=["PTL_NAME"])
                phase_op_group = pd.DataFrame(
                    {
                        "PHASE": np.array([0] * total_ptl),
                        "OPTGROUP": np.array([1] * total_ptl),
                        "GROUPNUM": np.array([1] * total_ptl),
                    }
                )
                randomsubset = pd.DataFrame(np.random.randint(1, high=3, size=total_ptl, dtype=int), columns=["RAND_SUBSET"])
                columns = [ptl_name, micrograph_coord, CTFs, phase_op_group, randomsubset]
            else:
                micrograph_coord = self.refinement[["FILM"]]

                columns = [micrograph_coord, CTFs]

            star_columns = pd.concat(columns, axis=1)

            star_header = data_optics_str + version + data_particles_header

            npvalue = star_columns.to_numpy(dtype=str, copy=True)
            np.savetxt(filename, npvalue, fmt='%s', header=star_header, delimiter="\t", comments='')
            logger.info(f"Metadata exported to {filename}")

        else:
            # run tomo export
            # tomo conversion
            tomo_data_header = """
# version 30001

data_global

loop_ 
_rlnTomoName #1 
_rlnTomoTiltSeriesName #2 
_rlnTomoFrameCount #3 
_rlnTomoSizeX #4 
_rlnTomoSizeY #5 
_rlnTomoSizeZ #6 
_rlnTomoHand #7 
_rlnOpticsGroupName #8 
_rlnTomoTiltSeriesPixelSize #9 
_rlnVoltage #10 
_rlnSphericalAberration #11 
_rlnAmplitudeContrast #12 
_rlnTomoImportFractionalDose #13 
"""
        
            # tomo tilt would be one block for each tilt series
            # tomo tilt data: data_{tilt_name}
            tomo_tilt_header = """
# version 30001

data_%s

loop_ 
_rlnTomoProjX #1 
_rlnTomoProjY #2 
_rlnTomoProjZ #3 
_rlnTomoProjW #4 
_rlnDefocusU #5 
_rlnDefocusV #6 
_rlnDefocusAngle #7 
_rlnCtfScalefactor #8 
_rlnMicrographPreExposure #9 
"""
            # subtomograms refinement star
            optics_header = """
# version 30001

data_optics

loop_ 
_rlnOpticsGroup #1 
_rlnOpticsGroupName #2 
_rlnSphericalAberration #3 
_rlnVoltage #4 
_rlnTomoTiltSeriesPixelSize #5 
_rlnCtfDataAreCtfPremultiplied #6 
_rlnImageDimensionality #7 
_rlnTomoSubtomogramBinning #8 
_rlnImagePixelSize #9 
_rlnImageSize #10
        """
            data_particles_header = """
# version 30001

data_particles

loop_ 
_rlnTomoName #1 
_rlnTomoParticleId #2 
_rlnTomoManifoldIndex #3 
_rlnCoordinateX #4 
_rlnCoordinateY #5 
_rlnCoordinateZ #6 
_rlnOriginXAngst #7 
_rlnOriginYAngst #8 
_rlnOriginZAngst #9 
_rlnAngleRot #10 
_rlnAngleTilt #11 
_rlnAnglePsi #12 
_rlnClassNumber #13 
_rlnRandomSubset #14 
_rlnTomoParticleName #15 
_rlnOpticsGroup #16 
_rlnImageName #17 
_rlnCtfImage #18 
_rlnGroupNumber #19 
_rlnNormCorrection #20 
_rlnLogLikeliContribution #21 
_rlnMaxValueProbDistribution #22 
_rlnNrOfSignificantSamples #23
        """
            particles_header = """
# version 30001 by xpytools

data_particles

loop_
_rlnTomoName #1
_rlnTomoParticleId #2
_rlnTomoManifoldIndex #3
_rlnCoordinateX #4
_rlnCoordinateY #5
_rlnCoordinateZ #6
_rlnOriginXAngst #7
_rlnOriginYAngst #8
_rlnOriginZAngst #9
_rlnAngleRot #10
_rlnAngleTilt #11
_rlnAnglePsi #12
_rlnClassNumber #13
_rlnRandomSubset #14
"""

            # global parameters
            pixel_size = self.scope_data["pixel_size"].values[0]
            voltage = self.scope_data["voltage"].values[0]
            cs = self.scope_data["CS"].values[0]
            ac = self.scope_data["AC"].values[0]
            dose_rate = self.scope_data["dose_rate"].values[0]

            dataset, format = os.path.splitext(Path(filename).name)
            tomogram_file = os.path.join( os.getcwd(), 'relion', f"{dataset}_tomograms{format}" )
            particle_file = os.path.join( os.getcwd(), 'relion', f"{dataset}_particles{format}" )

            EXTEND_START = 16

            FILM_COL = 8 - 1
            SCANORD_COL = 20 - 1
            PTLIND_COL = 17 - 1

            TILTAN_COL = 18 - 1
            NOMRX_COL = 24 - 1
            NORMZ_COL = 26 - 1 
            MATRIX0_COL = 27 - 1
            MATRIX12_COL = 39 - 1
            MATRIX13_COL = 40 - 1
            MATRIX15_COL = 42 - 1
            PPSI_COL = 43 - 1 
            PPHI_COL = 45 - 1

            # 2 star files are required: 
            # tomogram.star (for ImportTomo), coord.star (for ImportParticle)

            #################
            # tomogram.star #
            #################

            header = tomo_data_header
            body = ""

            logger.info(f"Exporting metadata for {len(self.data)} tomograms")
            with tqdm(desc="Progress", total=len(self.data), file=TQDMLogger()) as pbar:
                for micrograph in self.data.keys():
                    film_index = imagelist.index(micrograph)
                    data = self.data[micrograph]

                    # link tilt-series .mrc to relion folder if needed
                    if not os.path.exists(os.path.join("relion", "Movies")):
                        os.mkdir(os.path.join("relion", "Movies"))
                    if not os.path.exists(os.path.join("relion", "Movies", f"{micrograph}.mrc")):
                        os.symlink(os.path.join(os.getcwd(), "mrc", f"{micrograph}.mrc"), os.path.join("relion", "Movies", f"{micrograph}.mrc"))

                    # raw image size
                    num_tilts = data["image"].values[0][-1]
                    x = data["image"].values[0][0]
                    y = data["image"].values[0][1]

                    tomo_x = data["tomo"].values[0][0]
                    tomo_y = data["tomo"].values[0][1]
                    # tomo_z = data["tomo"].values[0][2]

                    # square, binning = getTomoBinFactor(x, y, bin_tomo_x=tomo_x)
                    binning = self.tomo_rec.loc["tomogram", "tomo_rec_binning"]
                    full_tomo_x = tomo_x * binning
                    full_tomo_y = tomo_y * binning
                    z = self.tomo_rec.loc["tomogram", "tomo_rec_thickness"]

                    # not sure what they are
                    hand = -1.0 if not self.micrograph_global["ctf_hand"].values[0] else 1.0
                    optic_group_name = "opticsGroup1"

                    micrograph_optics = list(map(str, [micrograph, f"Movies/{micrograph}.mrc", num_tilts, full_tomo_x, full_tomo_y, z, hand, optic_group_name, pixel_size, voltage, cs, ac, dose_rate]))
                    header += "\t".join(micrograph_optics)
                    header += "\n"

                    body += "\n\n"
                    body += tomo_tilt_header % (micrograph)
                    for tilt in range(num_tilts):

                        tilt_angle = data["tlt"].values[tilt][0]
                        xf = data["ali"].values[tilt]

                        df1 = data["ctf"].values[tilt][1]
                        df2 = data["ctf"].values[tilt][2]
                        astang = data["ctf"].values[tilt][3]
                        ctf_scale = 1.0 
                        scanord = data["order"].values[tilt][0]
                        exposure = scanord * dose_rate

                        # add csp tilt parameters to xf
                        dx_tilt = dy_tilt = 0.0

                        """
                        condition = np.where(
                                        (self.refinement.values[:, FILM_COL] == film_index) & \
                                        (self.extended.values[:, SCANORD_COL - EXTEND_START] == scanord)
                                    ) 
                        if condition[0].size != 0:
                            tilt_data = self.extended.values[condition, :][0, 0, :]
                            dx_tilt, dy_tilt = tilt_data[MATRIX12_COL - EXTEND_START : MATRIX13_COL - EXTEND_START + 1]
                            tilt_angle = tilt_data[TILTAN_COL - EXTEND_START]
                        """

                        xf[4] -= dx_tilt / pixel_size
                        xf[5] -= dy_tilt / pixel_size

                        matrix = getRelionMatrix(tilt_angle, xf, z, [x, y], full_tomo_x, full_tomo_y)
                        for r in range(matrix.shape[0]):
                            body += f"[{matrix[r,0]:.10f},{matrix[r,1]:.10f},{matrix[r,2]:.10f},{matrix[r,3]:.10f}] "


                        micrograph_tilt = list(map(str, [df1, df2, astang, ctf_scale, exposure]))
                        body += "\t".join(micrograph_tilt) + "\n"

                    pbar.update(1)

            header += body
            with open(tomogram_file, "w") as f:
                f.write(header)
            logger.info(f"Tomogram metadata exported to {tomogram_file}")

            ###### end of tomogram.star #######

            if coords:
                #################
                #   coord.star  #
                #################
                header = particles_header

                counter = 1
                manifold = 1 
                class_num = 1
                random_subset = 2

                logger.info(f"Exporting particle metadata from {len(self.data)} tomograms")
                with tqdm(desc="Progress", total=len(self.data), file=TQDMLogger()) as pbar:
                    for micrograph in self.data.keys():
                        film_index = imagelist.index(micrograph)
                        data = self.data[micrograph]
                        x = data["image"].values[0][0]
                        y = data["image"].values[0][1]
                        tomo_x = data["tomo"].values[0][0]
                        tomo_y = data["tomo"].values[0][1]
                        tomo_z = data["tomo"].values[0][2]

                        binning = self.tomo_rec.loc["tomogram", "tomo_rec_binning"]
                        full_tomo_x = tomo_x * binning
                        full_tomo_y = tomo_y * binning
                        full_thickness = self.tomo_rec.loc["tomogram", "tomo_rec_thickness"]

                        coordinates = data["box"].values
                        for particle_index, coord in enumerate(coordinates):
                            x, y, z = coord
                            relion_x, relion_y, relion_z = spk2Relion(x, y, z, binning, full_tomo_x, full_tomo_y, thickness=full_thickness, tomo_x_bin=tomo_x, tomo_y_bin=tomo_y, tomo_z_bin=tomo_z)

                            """
                            # try different scanning orders to get particle alignment 
                            # NOTE: ensure ptlind should match the index in spk file
                            for scanord in range(0, 40, 5):
                                condition = np.where(
                                                (self.refinement.values[:, FILM_COL] == film_index) & \
                                                (self.extended.values[:, PTLIND_COL - EXTEND_START] == particle_index) & \
                                                (self.extended.values[:, SCANORD_COL - EXTEND_START] == scanord)
                                            )
                                if condition[0].size != 0:
                                    break 

                            # if particle is not in the parfile 
                            if condition[0].size == 0:
                                continue

                            particle_data = self.extended.values[condition, :][0, 0, :]

                            matrix = particle_data[MATRIX0_COL - EXTEND_START : MATRIX15_COL - EXTEND_START + 1]
                            ppsi, ptheta, pphi = particle_data[PPSI_COL - EXTEND_START : PPHI_COL - EXTEND_START + 1]
                            normX, normY, normZ = particle_data[NOMRX_COL - EXTEND_START : NORMZ_COL - EXTEND_START + 1]
                            rot, tilt, psi, dx, dy, dz = alignment2Relion(matrix, ppsi, ptheta, pphi, normX, normY, normZ)

                            # relion will reset translation to zero during importing particles
                            # so we add the translation to coordinates
                            relion_x -= dx / pixel_size
                            relion_y -= dy / pixel_size
                            relion_z -= dz / pixel_size
                            dx = dy = dz = 0.0
                            """
                            # there is no alignment info here, only particle coordinates
                            particle = list(map(str, [micrograph, counter, manifold, relion_x, relion_y, relion_z, f"{0:.3f}", f"{0:.3f}", f"{0:.3f}", f"{0:.2f}", f"{tilt:.2f}", f"{0:.2f}", class_num, random_subset]))
                            header += "\t".join(particle) + "\n"

                            counter += 1
                            class_num *= -1
                            random_subset = 1 if random_subset == 2 else 2
                        pbar.update(1)

                with open(particle_file, "w") as f:
                    f.write(header)
                logger.info(f"Particle metadata exported to {particle_file}")

                ###### end of coord.star #######


    def SpaStar2meta(self, refinestar, motionstar, rln_path="relion", linkavg=True):
        """
        Extract metadata from RELION SPA star files
        """
        from_motion = os.path.isfile(motionstar)
        if from_motion:
            motion_corr = parse_star_tables(motionstar)

            self.scope_data.at[0, "pixel_size"] = motion_corr[Relion.OPTICDATA][Relion.MICROGRAPHORIGINALPIXELSIZE].values[0]
            self.scope_data["voltage"] = motion_corr[Relion.OPTICDATA][Relion.VOLTAGE].values[0]
            self.scope_data["AC"] = motion_corr[Relion.OPTICDATA][Relion.AC].values[0]
            self.scope_data["CS"] = motion_corr[Relion.OPTICDATA][Relion.CS].values[0]
            self.micrograph_global.at[0, "image_pixel_size"] =  motion_corr[Relion.OPTICDATA][Relion.MICROGRAPHPIXELSIZE].values[0]

        refinement_meta = parse_star_tables(refinestar)

        # get drifts from motion correction star
        assert Relion.PARTICLEDATA in refinement_meta.keys(), "No particle information from star file"
        self.refinement=refinement_meta[Relion.PARTICLEDATA]
        self.mode="spr"
        refinement_opt = refinement_meta[Relion.OPTICDATA]

        if Relion.MICROGRAPH_NAME in self.refinement.columns:
            rlnlist = self.refinement[Relion.MICROGRAPH_NAME].apply(os.path.basename).values
            imagelist = np.unique(rlnlist) # remove duplicates and keep the order
        # get alignment and ctf metadta from star
        if linkavg:
            relion_avgs = self.refinement[Relion.MICROGRAPH_NAME].values
            avgs_array = np.unique(relion_avgs)
            logger.info(f"Generating symbolic links to raw data for {len(avgs_array)} micrographs")
            with tqdm(desc="Progress", total=len(avgs_array), file=TQDMLogger()) as pbar:
                for avg in avgs_array:
                    avg_src = os.path.join(rln_path, avg)
                    dst_name = os.path.join("mrc", os.path.basename(avg))
                    # update image size - from .mrc images
                    assert (os.path.exists(avg_src)), f"{avg_src} does not exist"
                    command = f"{get_imod_path()}/bin/header -size '{avg_src}'"
                    [output, error] = run_shell_command(command, verbose=False)
                    x, y, z = list(map(int, output.split()))
                    arr = np.array([[x, y, z]])
                    imagekey = os.path.basename(avg).replace(".mrc", "")
                    if imagekey not in self.data:
                        self.data[imagekey] = {}
                    self.data[imagekey]["image"] = pd.DataFrame(arr, columns=FILES_TOMO["image"]["header"])
                    symlink_force(avg_src, dst_name)

                    pbar.update(1)

        if from_motion:

            logger.info(f"Importing CTF parameters and particle metadata from {len(imagelist)} movies")
            with tqdm(desc="Progress", total=len(imagelist), file=TQDMLogger()) as pbar:
                for image in imagelist:
                    imagekey = image.replace(".mrc", "")
                    check_images = motion_corr[Relion.MICROGRAPHDATA][Relion.MICROGRAPH_META].str.contains(imagekey)
                    if check_images.any():
                        motionstar =  motion_corr[Relion.MICROGRAPHDATA][Relion.MICROGRAPH_META][check_images.values].values[0]
                        image_general, frame_motions =  Read_MotionCorr(os.path.join(rln_path, motionstar))

                        if image == imagelist[0]:
                            self.UpdateGeneral(image_general, fromstar=True)
                            self.UpdatePtlOptics(refinement_opt, fromstar=True)

                        self.UpdateDrift(imagekey, frame_motions, fromstar=True)

                        datablock = self.refinement.loc[self.refinement[Relion.MICROGRAPH_NAME].str.contains(image)]
                        ctfs = []
                        for ctfparameter in Relion.CTF_PARAMS:
                            if ctfparameter in datablock.columns:
                                ctfs.append(ctfparameter)
                        ctfmeta = datablock[ctfs]
                        self.UpdateCTF(imagekey, ctfmeta, fromstar=True)

                        coords = datablock[Relion.COORDS].to_numpy(dtype=float)

                        self.UpdateCoord(imagekey, coords, fromstar=True)
                    pbar.update(1)

        else:

            logger.info(f"Importing CTF parameters and particle metadata from {len(imagelist)} micrographs")
            with tqdm(desc="Progress", total=len(imagelist), file=TQDMLogger()) as pbar:
                for image in imagelist:
                    imagekey = image.replace(".mrc", "")
                    datablock = self.refinement.loc[self.refinement[Relion.MICROGRAPH_NAME].str.contains(image)]
                    ctfs = []
                    for ctfparameter in Relion.CTF_PARAMS:
                        if ctfparameter in datablock.columns:
                            ctfs.append(ctfparameter)
                    ctfmeta = datablock[ctfs]
                    self.UpdateCTF(imagekey, ctfmeta, fromstar=True)

                    coords = datablock[Relion.COORDS].to_numpy(dtype=float)

                    self.UpdateCoord(imagekey, coords, fromstar=True)
                    frame_motions = np.array([[0, 0]])
                    self.UpdateDrift(imagekey, frame_motions, fromstar=True)

                    pbar.update(1)

            return list(self.data.keys())


    def TomoStar2meta(self, tomostar, refinestar, rln_path="relion/"):
        """ Convert TOMO metadata from Relion to PYP 

            Rquired inputs (Relion): 
                - tomograms.star 
                - refinement.star (any star file from Refine3D)
                - *.tlt
                - *.xf
            Outputs (PYP):
                - Metadata in pickle 
                - refinement.txt (sub-volumne averaging) 

                NOTE: You'll need to run the first iteration of csp to obtain initial parfile 
                        csp -refine_iter 2 -refine_par=refinement.txt 

        Args:
            tomostar (Path): Path to tomograms.star
            refinestar (Path): Path to run_iter???_data.star
            rln_path (str, optional): Path to Relion project folder, used for searching all .tlt and .xf files. Defaults to "relion/".

        Returns:
            Dict: Metadata for all the tilt-series 
        """

        assert (os.path.exists(tomostar)), f"{tomostar} does not exist"
        assert (os.path.exists(refinestar)), f"{refinestar} does not exist"

        #################
        # tomogram.star #
        #################
        tomogram_star = parse_star_tables(tomostar)
        tomogram_star_keys = set(tomogram_star.keys()) - set("data_global")
        tomograms = {}

        logger.info(f"Generating symbolic links to raw data for {tomogram_star['data_global'].size()} tilt-series")
        with tqdm(desc="Progress", total=tomogram_star["data_global"].size(), file=TQDMLogger()) as pbar:

            for index, row in tomogram_star["data_global"].iterrows():
                name = row[Relion.TOMONAME]
                path = row[Relion.TILTSERIESPATH]
                numTilt = row[Relion.TILT]
                x = row[Relion.TOMOX]
                y = row[Relion.TOMOY]
                z = row[Relion.TOMOZ]
                # store relion tomogram size x, y, z for further coordinate conversion
                tomograms[name] = [x, y, z, os.path.dirname(path)] 

                if name not in self.data:
                    self.data[name] = {} 

                # assume pixel size, cs, voltage etc. are the same 
                self.scope_data["voltage"] = row[Relion.VOLTAGE]
                self.scope_data["AC"] = row[Relion.AC]
                self.scope_data["CS"] = row[Relion.CS]
                self.scope_data["pixel_size"] = row[Relion.TOMOPIXELSIZE]
                self.scope_data["dose_rate"] = row[Relion.TOMODOSE]
                self.micrograph_global["ctf_hand"] = row[Relion.HAND]

                # update image size - from .mrc images
                assert (os.path.exists(Path(rln_path) / path)), f"{Path(rln_path) / path} does not exist"

                command = f"{get_imod_path()}/bin/header -size '{Path(rln_path) / path}'"
                [output, error] = run_shell_command(command, verbose=False)
                x, y, z = list(map(int, output.split()))
                arr = np.array([[x, y, z]])
                self.data[name]["image"] = pd.DataFrame(arr, columns=FILES_TOMO["image"]["header"])
                symlink_force(f"{Path(rln_path) / path}", f"{Path('mrc') / f'{name}.mrc'}")

                pbar.update(1)

        logger.info(f"Importing tomography metadata from {self.data.size()} tilt-series")
        with tqdm(desc="Progress", total=self.data.size(), file=TQDMLogger()) as pbar:
            for tomo in self.data.keys():

                tag = f"data_{tomo}"
                assert (tag in tomogram_star_keys), f"{tomo} not in star file"

                movie_dir = tomograms[tomo][3]

                # tilt - from .tlt file
                tlt_file = Path(rln_path) / movie_dir / f"{tomo}.tlt"
                assert (os.path.exists(tlt_file)), f"{tomo}.tlt not found. Please put it in {Path(rln_path) / movie_dir}"
                tlt = np.loadtxt(tlt_file, ndmin=1, comments=HEADERS, dtype=float)
                df = pd.DataFrame(np.reshape(tlt, (tlt.shape[0], 1)), index=FILES_TOMO["tlt"]["index"], columns=FILES_TOMO["tlt"]["header"])
                self.data[tomo]["tlt"] = df

                # ali - from .xf file
                xf_file = Path(rln_path) / movie_dir / f"{tomo}.xf"
                assert (os.path.exists(xf_file)), f"{tomo}.xf not found. Please put it in {Path(rln_path) / movie_dir}"
                ali = np.loadtxt(xf_file, ndmin=2, comments=HEADERS, dtype=float)
                df = pd.DataFrame(ali, index=FILES_TOMO["ali"]["index"], columns=FILES_TOMO["ali"]["header"])
                self.data[tomo]["ali"] = df

                # get shift info from IMOD tilt.com script for tomogram reconstruction 
                imod_tilt_file = Path(rln_path) / movie_dir / f"{tomo}.com"
                if os.path.exists(imod_tilt_file):
                    with open(imod_tilt_file) as f:
                        try:
                            tomogram_shifts = [line for line in f.read().split("\n") if "SHIFT" in line][0]
                            shiftx, shiftz = tomogram_shifts.split()[1:]
                            tomograms[tomo].append([float(shiftx), float(shiftz)]) 
                        except:
                            logger.warning(f"SHIFT not found in {tomo}.com. Assume no shifts applied to tomogram in RELION")
                            tomograms[tomo].append([0.0, 0.0])
                else:
                    tomograms[tomo].append([0.0, 0.0]) 
                    logger.warning(f"{tomo}.com not found. We'll assume you do NOT shift your tomogram in RELION")

                # ctf - from tomograms.star
                df1 = tomogram_star[tag][Relion.DEFOCUSU].to_numpy()
                df2 = tomogram_star[tag][Relion.DEFOCUSV].to_numpy()
                astang = tomogram_star[tag][Relion.DEFOCUSANGLE].to_numpy()
                cc = np.copy(astang)
                est_res = np.copy(astang)
                cc.fill(0.1)       # make up some values
                est_res.fill(5.0)  # make up some values

                # order - derived from dose fraction in tomograms.star
                exposures = tomogram_star[tag][Relion.MICROGRAPHPREEXPOSURE].to_numpy()
                scanord = exposures / self.scope_data["dose_rate"].values[0]

                # update dictionary using DataFrame
                ctf = np.row_stack((tlt, df1, df2, astang, cc, est_res)).T
                self.data[tomo]["ctf"] = pd.DataFrame(ctf, index=FILES_TOMO["ctf"]["index"], columns=FILES_TOMO["ctf"]["header"])
                self.data[tomo]["order"] = pd.DataFrame(scanord, index=FILES_TOMO["order"]["index"], columns=FILES_TOMO["order"]["header"])

                tomo_size = np.array([[512,512,256]]) # asign default values, change during pyp processing if needed
                self.data[tomo]["tomo"] = pd.DataFrame(tomo_size, index=FILES_TOMO["tomo"]["index"], columns=FILES_TOMO["tomo"]["header"])
                self.data[tomo]["box"] = []

                pbar.update(1)

        ##################
        # particles.star #
        ##################
        particle_star = parse_star_tables(refinestar)

        counter = 1
        tomo_spike_counter = {}
        refinement_header = """number  lwedge  uwedge  posX    posY    posZ    geomX   geomY   geomZ   normalX normalY normalZ matrix[0]       matrix[1]       matrix[2]        matrix[3]       matrix[4]       matrix[5]       matrix[6]       matrix[7]       matrix[8]       matrix[9]       matrix[10]       matrix[11]      matrix[12]      matrix[13]      matrix[14]      matrix[15]      magnification[0]       magnification[1]      magnification[2]        cutOffset       filename\n"""
        refinement = ["" for _ in range(len(particle_star["data_particles"].index))]

        logger.info(f"Importing particle metadata from {particle_star['data_particles'].size()} tilt-series")
        with tqdm(desc="Progress", total=particle_star["data_particles"].size(), file=TQDMLogger()) as pbar:
            for index, row in particle_star["data_particles"].iterrows():
                name = row[Relion.TOMONAME]
                x, y = self.data[name]["image"].at[0, "x"], self.data[name]["image"].at[0, "y"]
                square, binning = getTomoBinFactor(x, y)

                coord_x, coord_y, coord_z = row[Relion.COORDX], row[Relion.COORDY], row[Relion.COORDZ]
                dx, dy, dz = row[Relion.ORIGINXANGST], row[Relion.ORIGINYANGST], row[Relion.ORIGINZANGST]
                rot, tilt, psi = row[Relion.ANGLEROT], row[Relion.ANGLETILT], row[Relion.ANGLEPSI]

                coord_x -= dx / self.scope_data["pixel_size"].values[0]
                coord_y -= dy / self.scope_data["pixel_size"].values[0]
                coord_z -= dz / self.scope_data["pixel_size"].values[0]

                if name not in tomo_spike_counter:
                    tomo_spike_counter[name] = 0

                # convert relion coordinate to PYP (.spk)
                coord_x, coord_y, coord_z = relion2Spk(coord_x, coord_y, coord_z, binning, tomograms[name][0], tomograms[name][1], tomograms[name][2], shiftx=tomograms[name][-1][0], shiftz=tomograms[name][-1][-1])
                self.data[name]["box"].append([coord_x, coord_y, coord_z])

                # translate right-handed ZYZ to left-handed ZXZ
                # first compile left-handedness matrix 
                mrot = vtk.rotation_matrix(np.radians(-rot), [0, 0, 1])
                mtilt = vtk.rotation_matrix(np.radians(-tilt), [0, 1, 0])
                mpsi = vtk.rotation_matrix(np.radians(-psi), [0, 0, 1])
                zyz = np.dot(mpsi, np.dot(mtilt, mrot))
                zxz = eulerZYZtoZXZ(zyz)

                # compile refinement.txt
                refinement[counter-1] = getTomoRefinement(name, zxz, self.data[name]["tlt"].values, tomo_spike_counter[name], counter)

                # increment global counter and local spike counter
                counter += 1
                tomo_spike_counter[name] += 1

                pbar.update(1)

        # wrap box coordinate in DataFrame
        logger.info(f"Converting particle metadata from {self.data.keys()} tilt-series")
        with tqdm(desc="Progress", total=self.data.size(), file=TQDMLogger()) as pbar:
            for tomo in self.data.keys():
                arr = np.asarray(self.data[tomo]["box"])
                self.data[tomo]["box"] = pd.DataFrame(arr, index=FILES_TOMO["box"]["index"], columns=FILES_TOMO["box"]["header"])

                pbar.update(1)

        with open("pyp_update_volumes.txt", "w") as f:
            f.write(refinement_header)
            f.write("\n".join(refinement))


        return list(self.data.keys())


    def star2par(self, starfile, mag, path="."):
        """
        frealign custom command star 2 par
        """

        parfile = os.path.basename(starfile).replace(".star", ".par")
        pardata = refinestar2pardata(starfile, mag=mag)

        ptl_num = pardata.shape[0]
        # film from 0
        # pardata[:, 7] = pardata[:, 7] - 1
        extended = np.zeros((ptl_num, 29))
        extended[:, [5, 10, 15, 20, 25]] = 1
        par = np.hstack((pardata, extended))

        # write parfile 
        frealign = frealign_parfile.Parameters(version="new", extended=True)
        frealign.write_parameter_file(
            os.path.join(path, parfile), par, parx=True, frealignx=False
        )
        self.refinement = pd.DataFrame(pardata, columns=PARHEADER)
        self.extended = pd.DataFrame(extended, columns=PAREXTENDED)
        self.mode = "spr"


def refinestar2pardata(starfile, mag=10000):
    """
    Convert refine3d star file to standard parfile, return np array
    """

    star_metadata = parse_star_tables(starfile)
    # optics = star_metadata[Relion.OPTICDATA]
    refinemeta = star_metadata[Relion.PARTICLEDATA]

    ptl_num = refinemeta.shape[0]
    initials = [100, -500, 1, 20, 0] # occ, logp, sigma, score, change
    stats = np.tile(initials, (ptl_num, 1))

    if Relion.MICROGRAPH_NAME in refinemeta.columns:
        imagelist = np.unique(refinemeta[Relion.MICROGRAPH_NAME].apply(os.path.basename).values)
        newfilm = refinemeta[Relion.MICROGRAPH_NAME].values

    else:
        imagelist = [0]

    # image name to FILM id
    if len(imagelist) > 1:
        for id, image in enumerate(imagelist):
            mask = refinemeta[Relion.MICROGRAPH_NAME].str.contains(image).to_numpy()
            newfilm = np.where(mask, id, newfilm)
        newfilm = newfilm.reshape(-1, 1)   
    else:
        newfilm = np.array([0] * ptl_num).reshape(-1, 1)
    
    if not all([ ctf in refinemeta.columns for ctf in Relion.CTF_PARAMS[:3]]):
        logger.error("Missing CTF information. Abort")
        sys.exit()
    ctf = refinemeta[Relion.CTF_PARAMS[:3]].to_numpy()

    alignment = Relion.ANGLES + Relion.ORIGINSANGST
    for align in alignment:
        if align not in refinemeta.columns:
            refinemeta[[align]] = pd.DataFrame(np.array([0]*ptl_num).reshape(-1, 1))
    
    angles = refinemeta[[Relion.ANGLEPSI, Relion.ANGLETILT, Relion.ANGLEROT]].to_numpy()
    shifts = refinemeta[[Relion.ORIGINXANGST, Relion.ORIGINYANGST]].to_numpy()
    shifts = - shifts

    pid = np.arange(1, ptl_num + 1).reshape(-1, 1)
    mag = np.array([mag] * ptl_num).reshape(-1, 1)

    pardata = np.hstack((pid, angles, shifts, mag, newfilm, ctf, stats))
    
    return pardata


def Read_MotionCorr(metastar):
    """
    Read micrograph metadata star from relion MotionCorrection job
    FOR special motion meta star, no loop_ in the table
    """
    with open(metastar, mode='r') as f:
        data0 = {}
        lines = f.read()
        tables = lines.split("data_")
        for t in tables:
            if "general" in t:
                content =[s for s in t.splitlines() if s]
                for item in content: 
                    if "_rln" in item:
                        data0.update({item.strip().split()[0].replace("_rln", "rln"): item.strip().split()[1]})
            elif "global_shift" in t:
                realdata = []
                columns = []
                content =[s.strip() for s in t.splitlines() if s]
                content.remove("global_shift")
                for item in content:
                    if "loop_" in item:
                        pass
                    elif "_rln" in item:
                        columns.append(item.strip().split()[0].replace("_rln", "rln"))
                    elif item and not "#" in item:
                        realdata.append(item.strip().split())  
            else:
                pass

        image_general = pd.DataFrame(data0, columns=data0.keys(), index=[0])
        fdata = pd.DataFrame(np.array(realdata), columns=columns)
        frame_motions = fdata[[Relion.MICROGRAPH_SHIFTX, Relion.MICROGRAPH_SHIFTY]].to_numpy()

    return image_general, frame_motions


def merge_par_selection(parfile, selected, parameters, merge_align=False):
    """
    Merge different classes after classification ready for tomoedit.
    selected: list that has the class numers 
    mergealign: whether merge selected alignment parameters or not
    """
    current_dir = os.getcwd()
    parfile1 = re.sub("_r[0-9][0-9]_", "_r%02d_" % selected[0], parfile)
    # check decompress
    if os.path.exists(parfile1) and parfile1.endswith(".bz2"):
        parfile1 = frealign_parfile.Parameters.decompress_parameter_file(parfile1, parameters["slurm_tasks"])
    elif not os.path.exists(parfile1):
        logger.error(f"Can't find corresponding parfiles: {parfile1}")
        sys.exit()
    pardata_keep1 = frealign_parfile.Parameters.from_file(parfile1).data
    
    n = pardata_keep1.shape[0]

    if len(selected) > 1:
        for k in selected[1:]:
            parfilek = re.sub("_r[0-9][0-9]_", "_r%02d_" % k, parfile)
            if os.path.exists(parfilek) and parfilek.endswith(".bz2"):
                    parfilek = frealign_parfile.Parameters.decompress_parameter_file(parfilek, parameters["slurm_tasks"])
            elif not os.path.exists(parfilek):
                logger.error("Can't find corresponding parfiles")
                sys.exit()

            if merge_align:
                pardatak = frealign_parfile.Parameters.from_file(parfilek).data
                mask = (pardatak[:, 11] >= parameters["reconstruct_min_occ"]).reshape(n, 1)
                pardata_keep1 = np.where(mask, pardatak, pardata_keep1)
            else:
                # read occ column only
                pardatak_occ = np.loadtxt(parfilek, usecols=11, comments="C")
                mask = pardatak_occ >= parameters["reconstruct_min_occ"]
                pardata_keep1[:,11] = np.where(mask, pardatak_occ, pardata_keep1[:, 11])

    occ_keepmask = pardata_keep1[:, 11] >= parameters["reconstruct_min_occ"]
    pardata_keep1[:, 11] = np.where(occ_keepmask, 100, 0)

    version = parameters["refine_metric"]
    frealign_par = frealign_parfile.Parameters(version=version)
    combined = "_K".join(str(x) for x in selected)
    parxfile = current_dir + "/frealign/maps/" + parameters["data_set"] + "_K" + combined + ".par"
    
    frealign_par.write_parameter_file(parxfile, pardata_keep1, parx=True, frealignx=False)
                
    # compress parfile
    parfile_name = os.path.basename(parxfile)
    os.chdir(current_dir + "/frealign/maps/")
    frealign_parfile.Parameters.compress_parameter_file(parfile_name, parfile_name.replace(".par", ".par.bz2"), parameters["slurm_tasks"])
    os.remove(parfile_name)
    os.chdir(current_dir)

    return parxfile.replace(".par", ".par.bz2")

##############

# Those functions from pyem by Daniel Asarnow, University of California, San Francisco
def parse_star_table(starfile, offset=0, nrows=None, keep_index=False):
    headers = []
    foundheader = False
    ln = 0
    with open(starfile, 'r') as f:
        f.seek(offset)
        for l in f:
            if l.lstrip().startswith("_"):
                foundheader = True
                lastheader = True
                if keep_index:
                    head = l.strip()
                else:
                    head = l.split('#')[0].strip().lstrip('_')
                headers.append(head)
            else:
                lastheader = False
            if foundheader and not lastheader:
                break
            ln += 1
        f.seek(offset)
        df = pd.read_csv(f, delimiter='\s+', header=None, skiprows=ln, nrows=nrows)
    df.columns = headers
    return df


def star_table_offsets(starfile):
    tables = {}
    with open(starfile) as f:
        l = f.readline()  # Current line
        ln = 0  # Current line number.
        offset = 0  # Char offset of current table.
        cnt = 0  # Number of tables.
        in_table = False  # True if file cursor is inside a table.
        in_loop = False
        blank_terminates = False
        while l:
            if l.lstrip().startswith("data"):
                table_name = l.strip()
                if in_table:
                    tables[table_name] = (offset, lineno, ln - 1, ln - data_line - 1)
                in_table = True
                in_loop = False
                blank_terminates = False
                offset = f.tell()  # Record byte offset of table.
                lineno = ln  # Record start line of table.
                cnt += 1  # Increment table count.
            if l.lstrip().startswith("loop"):
                in_loop = True
            elif in_loop and not l.startswith("_"):
                in_loop = False
                blank_terminates = True
                data_line = ln
            if blank_terminates and in_table and l.isspace():  # Allow blankline to terminate table.
                in_table = False
                tables[table_name] = (offset, lineno, ln - 1, ln - data_line)
            l = f.readline()  # Read next line.
            ln += 1  # Increment line number.
        if in_table and table_name not in tables:
            tables[table_name] = (offset, lineno, ln, ln - data_line)
        return tables


def augment_star_ucsf(df, inplace=True):
    df = df if inplace else df.copy()
    df.reset_index(inplace=True)
    if Relion.IMAGE_NAME in df:
        df[UCSF.IMAGE_INDEX], df[UCSF.IMAGE_PATH] = \
            df[Relion.IMAGE_NAME].str.split("@").str
        df[UCSF.IMAGE_INDEX] = pd.to_numeric(df[UCSF.IMAGE_INDEX]) - 1

        if Relion.IMAGE_ORIGINAL_NAME not in df:
            df[Relion.IMAGE_ORIGINAL_NAME] = df[Relion.IMAGE_NAME]

    if Relion.IMAGE_ORIGINAL_NAME in df:
        df[UCSF.IMAGE_ORIGINAL_INDEX], df[UCSF.IMAGE_ORIGINAL_PATH] = \
            df[Relion.IMAGE_ORIGINAL_NAME].str.split("@").str
        df[UCSF.IMAGE_ORIGINAL_INDEX] = pd.to_numeric(df[UCSF.IMAGE_ORIGINAL_INDEX]) - 1

    if UCSF.IMAGE_PATH in df:
        df[UCSF.IMAGE_BASENAME] = df[UCSF.IMAGE_PATH].apply(os.path.basename)

    if UCSF.IMAGE_ORIGINAL_PATH in df:
        df[UCSF.IMAGE_ORIGINAL_BASENAME] = df[UCSF.IMAGE_ORIGINAL_PATH].apply(os.path.basename)

    if Relion.MICROGRAPH_NAME in df:
        df[UCSF.MICROGRAPH_BASENAME] = df[Relion.MICROGRAPH_NAME].apply(os.path.basename)
    return df


def parse_star(starfile, keep_index=False, augment=True, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
        for t in tables}
    if Relion.OPTICDATA in dfs:
        if Relion.PARTICLEDATA in dfs:
            data_table = Relion.PARTICLEDATA
        elif Relion.MICROGRAPHDATA in dfs:
            data_table = Relion.MICROGRAPHDATA
        elif Relion.IMAGEDATA in dfs:
            data_table = Relion.IMAGEDATA
        else:
            data_table = None
        if data_table is not None:
            df = pd.merge(dfs[Relion.OPTICDATA], dfs[data_table], on=Relion.OPTICSGROUP)
        else:
            df = dfs[Relion.OPTICDATA]
    else:
        df = dfs[next(iter(dfs))]
    df = check_defaults(df, inplace=True)
    if augment:
        augment_star_ucsf(df, inplace=True)
    return df


def check_defaults(df, inplace=False):
    df = df if inplace else df.copy()
    if Relion.PHASESHIFT not in df:
        df[Relion.PHASESHIFT] = 0

    if Relion.IMAGEPIXELSIZE in df:
        if Relion.DETECTORPIXELSIZE not in df and Relion.MAGNIFICATION not in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.IMAGEPIXELSIZE]
            df[Relion.MAGNIFICATION] = 10000
        elif Relion.DETECTORPIXELSIZE in df:
            df[Relion.MAGNIFICATION] = df[Relion.DETECTORPIXELSIZE] / df[Relion.IMAGEPIXELSIZE] * 10000
        elif Relion.MAGNIFICATION in df:
            df[Relion.DETECTORPIXELSIZE] = df[Relion.MAGNIFICATION] * df[Relion.IMAGEPIXELSIZE] / 10000
    elif Relion.DETECTORPIXELSIZE in df and Relion.MAGNIFICATION in df:
        df[Relion.IMAGEPIXELSIZE] = df[Relion.DETECTORPIXELSIZE] * df[Relion.MAGNIFICATION] / 10000

    for it in zip(Relion.ORIGINSANGST3D, Relion.ORIGINS3D):
        if it[0] in df:
            df[it[1]] = df[it[0]] / df[Relion.IMAGEPIXELSIZE]
        elif it[1] in df:
            df[it[0]] = df[it[1]] * df[Relion.IMAGEPIXELSIZE]

    if Relion.ORIGINZANGST in df:
        df[Relion.IMAGEDIMENSION] = 3
    else:
        df[Relion.IMAGEDIMENSION] = 2

    if Relion.OPTICSGROUPNAME in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.OPTICSGROUPNAME].astype('category').cat.codes

    if Relion.BEAMTILTCLASS in df and Relion.OPTICSGROUP not in df:
        df[Relion.OPTICSGROUP] = df[Relion.BEAMTILTCLASS]
    return df


def parse_star_tables(starfile, keep_index=False, nrows=sys.maxsize):
    tables = star_table_offsets(starfile)
    dfs = {t: parse_star_table(starfile, offset=tables[t][0], nrows=min(tables[t][3], nrows), keep_index=keep_index)
        for t in tables}
    return dfs


# relion class from pyem by Daniel Asarnow, University of California, San Francisco
class Relion:
    # Relion 2+ fields.
    MICROGRAPH_NAME = "rlnMicrographName"
    MICROGRAPH_NAME_NODW = "rlnMicrographNameNoDW"
    IMAGE_NAME = "rlnImageName"
    IMAGE_ORIGINAL_NAME = "rlnImageOriginalName"
    RECONSTRUCT_IMAGE_NAME = "rlnReconstructImageName"
    COORDX = "rlnCoordinateX"
    COORDY = "rlnCoordinateY"
    COORDZ = "rlnCoordinateZ"
    ORIGINX = "rlnOriginX"
    ORIGINY = "rlnOriginY"
    ORIGINZ = "rlnOriginZ"
    ANGLEROT = "rlnAngleRot"
    ANGLETILT = "rlnAngleTilt"
    ANGLEPSI = "rlnAnglePsi"
    CLASS = "rlnClassNumber"
    DEFOCUSU = "rlnDefocusU"
    DEFOCUSV = "rlnDefocusV"
    DEFOCUS = [DEFOCUSU, DEFOCUSV]
    DEFOCUSANGLE = "rlnDefocusAngle"
    CS = "rlnSphericalAberration"
    PHASESHIFT = "rlnPhaseShift"
    AC = "rlnAmplitudeContrast"
    VOLTAGE = "rlnVoltage"
    MAGNIFICATION = "rlnMagnification"
    DETECTORPIXELSIZE = "rlnDetectorPixelSize"
    BEAMTILTX = "rlnBeamTiltX"
    BEAMTILTY = "rlnBeamTiltY"
    BEAMTILTCLASS = "rlnBeamTiltClass"
    CTFSCALEFACTOR = "rlnCtfScalefactor"
    CTFBFACTOR = "rlnCtfBfactor"
    CTFMAXRESOLUTION = "rlnCtfMaxResolution"
    CTFFIGUREOFMERIT = "rlnCtfFigureOfMerit"
    GROUPNUMBER = "rlnGroupNumber"
    RANDOMSUBSET = "rlnRandomSubset"
    AUTOPICKFIGUREOFMERIT = "rlnAutopickFigureOfMerit"

    # Relion 3 fields.
    OPTICSGROUP = "rlnOpticsGroup"
    OPTICSGROUPNAME = "rlnOpticsGroupName"
    ODDZERNIKE = "rlnOddZernike"
    EVENZERNIKE = "rlnEvenZernike"
    MAGMAT00 = "rlnMagMat00"
    MAGMAT01 = "rlnMagMat01"
    MAGMAT10 = "rlnMagMat10"
    MAGMAT11 = "rlnMagMat11"
    IMAGEPIXELSIZE = "rlnImagePixelSize"
    IMAGESIZE = "rlnImageSize"
    IMAGESIZEX = "rlnImageSizeX"
    IMAGESIZEY = "rlnImageSizeY"
    IMAGESIZEZ = "rlnImageSizeZ"
    IMAGEDIMENSION = "rlnImageDimensionality"
    ORIGINXANGST = "rlnOriginXAngst"
    ORIGINYANGST = "rlnOriginYAngst"
    ORIGINZANGST = "rlnOriginZAngst"
    MICROGRAPHPIXELSIZE = "rlnMicrographPixelSize"
    MICROGRAPHORIGINALPIXELSIZE = "rlnMicrographOriginalPixelSize"
    MICROGRAPHBIN = "rlnMicrographBinning"
    MICROGRAPHDOSERATE = "rlnMicrographDoseRate"
    MICROGRAPHPREEXPOSURE = "rlnMicrographPreExposure"
    MICROGRAPHSTARTFRAME = "rlnMicrographStartFrame"
    MTFFILENAME = "rlnMtfFileName"
    MICROGRAPH_META = "rlnMicrographMetadata"
    MICROGRAPH_FRAMENUMBER = "rlnMicrographFrameNumber"
    MICROGRAPH_SHIFTX = "rlnMicrographShiftX"
    MICROGRAPH_SHIFTY = "rlnMicrographShiftY"


    # Tomo 
    TOMONAME = "rlnTomoName"
    TILTSERIESPATH = "rlnTomoTiltSeriesName"
    TILT = "rlnTomoFrameCount"
    TOMOX = "rlnTomoSizeX"
    TOMOY = "rlnTomoSizeY"
    TOMOZ = "rlnTomoSizeZ"
    HAND = "rlnTomoHand"
    TOMOPIXELSIZE = "rlnTomoTiltSeriesPixelSize"
    TOMODOSE = "rlnTomoImportFractionalDose"



    # Field lists.
    COORDS = [COORDX, COORDY]
    ORIGINS = [ORIGINX, ORIGINY]
    ORIGINS3D = [ORIGINX, ORIGINY, ORIGINZ]
    ORIGINSANGST = [ORIGINXANGST, ORIGINYANGST]
    ORIGINSANGST3D = [ORIGINXANGST, ORIGINYANGST, ORIGINZANGST]
    ANGLES = [ANGLEROT, ANGLETILT, ANGLEPSI]
    ALIGNMENTS = ANGLES + ORIGINS3D + ORIGINSANGST3D
    CTF_PARAMS = [DEFOCUSU, DEFOCUSV, DEFOCUSANGLE, CS, PHASESHIFT, AC,
                  BEAMTILTX, BEAMTILTY, BEAMTILTCLASS, CTFSCALEFACTOR, CTFBFACTOR,
                  CTFMAXRESOLUTION, CTFFIGUREOFMERIT]
    MICROSCOPE_PARAMS = [VOLTAGE, MAGNIFICATION, DETECTORPIXELSIZE]
    MICROGRAPH_COORDS = [MICROGRAPH_NAME] + COORDS
    PICK_PARAMS = MICROGRAPH_COORDS + [ANGLEPSI, CLASS, AUTOPICKFIGUREOFMERIT]

    FIELD_ORDER = [IMAGE_NAME, IMAGE_ORIGINAL_NAME, MICROGRAPH_NAME, MICROGRAPH_NAME_NODW] + \
                   COORDS + ALIGNMENTS + MICROSCOPE_PARAMS + CTF_PARAMS + \
                  [CLASS + GROUPNUMBER + RANDOMSUBSET + OPTICSGROUP]

    RELION2 = ORIGINS3D + [MAGNIFICATION, DETECTORPIXELSIZE]

    RELION30 = [BEAMTILTCLASS]

    RELION31 = ORIGINSANGST3D + [BEAMTILTX, BEAMTILTY, OPTICSGROUP, OPTICSGROUPNAME,
                ODDZERNIKE, EVENZERNIKE, MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11,
                IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    OPTICSGROUPTABLE = [AC, CS, VOLTAGE, BEAMTILTX, BEAMTILTY, OPTICSGROUPNAME, ODDZERNIKE, EVENZERNIKE,
                        MAGMAT00, MAGMAT01, MAGMAT10, MAGMAT11, IMAGEPIXELSIZE, IMAGESIZE, IMAGEDIMENSION]

    # Data tables.
    OPTICDATA = "data_optics"
    MICROGRAPHDATA = "data_micrographs"
    PARTICLEDATA = "data_particles"
    IMAGEDATA = "data_images"
    SHIFTDATA = "data_global_shift"
    MICROGRAPH_GENERAL = "data_general"

    # Data type specification.
    DATATYPES = {OPTICSGROUP: int}


class UCSF:
    IMAGE_PATH = "ucsfImagePath"
    IMAGE_BASENAME = "ucsfImageBasename"
    IMAGE_INDEX = "ucsfImageIndex"
    IMAGE_ORIGINAL_PATH = "ucsfImageOriginalPath"
    IMAGE_ORIGINAL_BASENAME = "ucsfImageOriginalBasename"
    IMAGE_ORIGINAL_INDEX = "ucsfImageOriginalIndex"
    MICROGRAPH_BASENAME = "ucsfMicrographBasename"
    UID = "ucsfUid"
    PARTICLE_UID = "ucsfParticleUid"
    MICROGRAPH_UID = "ucsfMicrographUid"
