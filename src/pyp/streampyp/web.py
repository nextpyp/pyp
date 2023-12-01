"""
talk to the website using JSON-RPC
https://jsonrpcclient.readthedocs.io/en/latest/api.html
"""

from asyncio.log import logger
import os
import sys
import re
from collections import namedtuple
from json import dumps as serialize
import pathlib

from jsonrpcclient.clients.http_client import HTTPClient
from jsonrpcclient.requests import Request


class Web:

    endpoint = "pyp"

    exists = "NEXTPYP_WEBHOST" in os.environ

    @classmethod
    def init_env(cls):
        # some pyp functions actually write to the home directory
        # but on a single-user web server that doesn't make sense anymore
        # so just put the "home" directory in the current directory
        os.environ["HOME"] = "./"

    def __init__(self):
        self.host = os.environ["NEXTPYP_WEBHOST"]
        self.token = os.environ["NEXTPYP_TOKEN"]
        self.client = HTTPClient("%s/%s" % (self.host, Web.endpoint))
        self.webid = os.environ["NEXTPYP_WEBID"]

    def _request(self, request):
        request["token"] = self.token
        response = self.client.send(request)
        return response.data.result

    def _request_raw(self, request):
        """
        Sends the JSON-RPC request, returns the raw text response from the server
        """
        request["token"] = self.token
        response = self.client.send_message(
            request=serialize(request), response_expected=True
        )
        return response.text

    def _send(self, message):
        """
        Sends the JSON-RPC request, and doesn't wait for a response
        """
        try:
            message["token"] = self.token
            self.client.send_message(
                request=serialize(message), response_expected=False
            )
        except:
            (ex_type, ex, traceback) = sys.exc_info()
            print(
                "\tfailed to send message to web server: %s: %s"
                % (ex_type.__name__, ex)
            )

    def ping(self):
        """
        :return: the string 'pong'
        """
        return self._request_raw(Request.ping())

    def slurm_started(self, arrayid):
        self._request(Request.slurm_started(webid=self.webid, arrayid=arrayid))

    def slurm_ended(self, arrayid, exit_code):
        self._request(Request.slurm_ended(webid=self.webid, arrayid=arrayid, exit_code=exit_code))

    def failed(self):
        """
        Attempts to send a failure signal to the website.
        Will not throw an exception, so safe to use inside error-handling blocks.
        """
        # get the array id, if any
        try:
            arrayid = int(os.environ["SLURM_ARRAY_TASK_ID"])
        except KeyError:
            arrayid = None
        try:
            self._request(Request.failed(webid=self.webid, arrayid=arrayid))
        except:
            pass

    def slurm_sbatch(
        self, web_name, cluster_name, commands, dir=None, env=[], args=[], deps=[], mpi=None
    ):
        """
        Launches a SLURM job using sbatch

        :param [str] web_name: A user-friendly name for the job, displayed on the website
        :param [str] cluster_name: The name of the job submitted to SLURM
        :param [str] commands: The bash commands to run, must be Web.CommandsScript or Web.CommandsGrid
        :param str dir: The directory in which to run the commands
        :param [(str, str)] env: A list of environment variables for the process, eg, [('VAR1', 'val'), ('VAR2', 'val')]
        :param [str] args: A list of arguments for sbatch, eg, ['--time=1']
        :param [str] deps: A list of job database IDs (returned from this function) to depend on, eg, ['cebm7FSpfDm3vly3']
        :param [dict] mpi: If not None, the website will launch the singularity container within mpirun.
                           Pass a dictionary of arguments to mpirun:
                             oversubscribe: boolean
                             cpus: int
        :return str: Returns the database id of the submitted job
        """

        # filter out empty arguments
        args = [it for it in args if len(it) > 0]

        # validate the arguments at submission time so we can fail early
        if any([arg.startswith("--output=") for arg in args]):
            raise ValueError(
                "job outputs are handled automatically by nextPYP, no need to specify them explicitly"
            )
        if any([arg.startswith("--error=") for arg in args]):
            raise ValueError(
                "job errors are handled automatically by nextPYP, no need to specify them explicitly"
            )
        if any([arg.startswith("--chdir=") for arg in args]):
            raise ValueError(
                "the submission directory is handled automatically by nextPYP, no need to specify it explicitly"
            )
        if any([arg.startswith("--array=") for arg in args]):
            raise ValueError(
                "the job array is handled automatically by nextPYP, no need to specify it explicitly"
            )

        return self._request(
            Request.slurm_sbatch(
                webid=self.webid,
                web_name=web_name,
                cluster_name=cluster_name,
                commands=commands.render(),
                dir=dir,
                env=env,
                args=args,
                deps=deps,
                mpi=mpi
            )
        )

    class CommandsScript:
        def __init__(self, commands, array_size=None, bundle_size=None):
            """
    		Runs commands together as a script inside a single launch of singularity/MPI

    		:param [str] commands: the list of commands
    		:param int: None to run commands sequentially on one SLURM node.
    		            Pass a positive integer to run the script in parallel on multiple SLURM nodes.
    		:param [int] bundle_size: The bundle size to use for SLURM jobs, or None to not use bundling
    		"""
            self.commands = commands
            self.array_size = array_size
            self.bundle_size = bundle_size

        def render(self):
            return {
                "type": "script",
                "commands": self.commands,
                "array_size": self.array_size,
                "bundle_size": self.bundle_size
            }

    class CommandsGrid:
        def __init__(self, commands, bundle_size=None):
            """
    		Runs commands as a 2D grid, some sequentially, and some in parallel.
    		Each command is individually wrapped in a singularity/MPI launch

    		:param [[str]] commands: The grid of commands.
    		                         The outer dimension is run in parallel as a SLURM array.
    		                         The inner dimension is run in sequence inside a single SLURM array element.
    		                         eg: [[c1, c2, c3]] runs three commands in the sequence c1,c2,c3 on a single compute node.
    		                             [[c1], [c2], [c3]] runs three commands in parallel, with each ci running on a different compute node.
    		                             [[c1, c2], [c3]] runs two commands c1,c2 on one compute node, and command c3 on another compute node
    		:param [int] bundle_size: The bundle size to use for SLURM jobs, or None to not use bundling
    		"""
            self.commands = commands
            self.bundle_size = bundle_size

        def render(self):
            return {
                "type": "grid",
                "commands": self.commands,
                "bundle_size": self.bundle_size
            }

    CTF = namedtuple(
        "CTF",
        [
            "mean_df",
            "cc",
            "df1",
            "df2",
            "angast",
            "ccc",
            "x",
            "y",
            "z",
            "pixel_size",
            "voltage",
            "magnification",
            "cccc",
            "counts",
        ],
    )

    AVGROT = namedtuple(
        "AVGROT",
        ["freq", "avgrot_noastig", "avgrot", "ctf_fit", "quality_fit", "noise"],
    )

    XF = namedtuple("XF", ["mat00", "mat01", "mat10", "mat11", "x", "y"])

    BOXX = namedtuple("BOXX", ["x", "y", "w", "h", "in_bounds", "cls"])

    REFINEMENT_DICTIONARY = {}

    def write_parameters(self, parameter_id, parameters):
        serial_parameters = parameters.copy()
        for k in serial_parameters:
            if isinstance(serial_parameters[k],pathlib.Path):
                serial_parameters[k] = str(serial_parameters[k])

        self._request(
            Request.write_parameters(
                webid=self.webid,
                parameter_id=parameter_id,
                parameters=serial_parameters,
            )
        )

    def write_micrograph(self, micrograph_id, ctf, avgrot, xf, boxx):
        self._request(
            Request.write_micrograph(
                webid=self.webid,
                micrograph_id=micrograph_id,
                ctf=ctf,
                avgrot=avgrot,
                xf=xf,
                boxx=boxx,
            )
        )

    def write_tiltseries(self, tiltseries_id, ctf, avgrot, xf, boxx, metadata):

        try:
            virion_coords = metadata["virion_coordinates"].tolist()
        except KeyError:
            virion_coords = None

        try:
            spike_coords = metadata["spike_coordinates"].tolist()
        except KeyError:
            spike_coords = None

        self._request(
            Request.write_tiltseries(
                webid=self.webid,
                tiltseries_id=tiltseries_id,
                ctf=ctf,
                avgrot=avgrot,
                xf=xf,
                boxx=boxx,
                metadata={
                    "tilts": metadata["tilts"],
                    "drift": [l.tolist() for l in metadata["drift"].values()],
                    "ctf_values": [l.tolist() for l in metadata["ctf_values"].values()],
                    "ctf_profiles": [l.tolist() for l in metadata["ctf_profiles"].values()],
                    "tilt_axis_angle": metadata["tilt_axis_angle"],
                    "virion_coordinates": virion_coords,
                    "spike_coordinates": spike_coords
                }
            )
        )

    def write_reconstruction(self, reconstruction_id, metadata, fsc, plots):

        series_information = re.search("\\d+_\\d+$", reconstruction_id)[0].split("_")
        class_num = int(series_information[0])
        iteration = int(series_information[1])

        self._request(
            Request.write_reconstruction(
                webid=self.webid,
                reconstruction_id=reconstruction_id,
                class_num=class_num,
                iteration=iteration,
                metadata=metadata,
                fsc=fsc.tolist(),
                plots=plots
            )
        )

    def write_refinement_bundle(self, refinement_bundle_id, iteration):

        self._request(
            Request.write_refinement_bundle(
                webid=self.webid,
                refinement_bundle_id=refinement_bundle_id,
                iteration=iteration
            )
        )

    def write_refinement(self, refinement_id, iteration):

        self._request(
            Request.write_refinement(
                webid=self.webid,
                refinement_id=refinement_id,
                iteration=iteration
            )
        )

    def write_classes(self, classes_id, metadata):

        self._request(
            Request.write_classes(
                webid=self.webid,
                classes_id=classes_id,
                metadata=metadata,
            )
        )

    def log(self, timestamp, level, path, line, msg):
        """
        Send a log entry to the website.

        :param int timestamp: milliseconds from the unix epoch
        :param int level: logging level numeric value
        :param str path: path to the originating python source file
        :param int line: line number in the originating python source file
        :param str msg: the message to log
        """
        return self._send(
            Request.log(
                webid=self.webid,
                timestamp=timestamp,
                level=level,
                path=path,
                line=line,
                msg=msg,
            )
        )
