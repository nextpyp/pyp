set -e # immediately exit on error
set -x # print commands to terminal

# ensure that we are at the tests folder of our app
cd "${0%/*}"

pwd

singularity --quiet --silent exec -B /nfs,/work,/scratch,/cifs,/hpc --no-home -B $HOME/.ssh -B $PYP_DIR:/opt/pyp pyp.sif pytest --slurm_mode --save_results
