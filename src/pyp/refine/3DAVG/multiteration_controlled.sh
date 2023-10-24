#!/bin/bash
# This file is multprotocol/iteration.sh
#
#SBATCH --job-name=stavg
#
# To execute: 
# qsub -v np=80,mydir=`pwd`,pattern=<pattern>,protocol=<protocol>[,mode=<startmode>,iter=<start iteration>,mailto=<email_address>,sendpng=<yes>] -l nodes=20:dc multiteration_controlled.sh
# program=MPI_Classification_2010_10_15_nothreads

module load MPICH/3.2.1
module load FFTW/3.3.7
program=${PYP_DIR}/external/TOMO2/asoft/bin/MPI_Classification

echo "Running on directory: $SLURM_SUBMIT_DIR"
echo "Using protocol: $protocol"
echo "Program is $program"

mydir=$SLURM_SUBMIT_DIR
cd $mydir

# mpdboot -f $PBS_NODEFILE -n `cat $PBS_NODEFILE | wc -l`

if [ ! -n "$iter" ]; then
    iter=1
fi
if [ ! -n "$mode" ]; then
    mode=0
fi
if [ ! -n "$liter" ]; then
    liter=7
fi
echo "starting at iteration: [${iter}]"
echo "initial mode is: [${mode}]"


# Run centering if mode equals 0
if [ $mode -eq 0 ]; then
    # Centering
    # mpiexec -n $SLURM_NTASKS $program `printf "${protocol}/iteration_%03d_mode_%d.xml"  ${iter} 0`
    srun -n $SLURM_NTASKS --mpi=pmi2 --export LD_LIBRARY_PATH $program `printf "${protocol}/iteration_%03d_mode_%d.xml"  ${iter} 0`
    rm *.tmp
    text=`printf "Finished centering for iteration %03d on directory %s" ${iter} ${mydir}`
    if [ -n $mailto ]; then
	echo $text | mutt -s "$text" $mailto
    fi
    while [ ! -e ./classify ]; do
        echo $text
	sleep 60
    done
    let mode=mode+1
fi

#
# LOOP THROUGH ITERATIONS
#

while [ $iter -le ${liter} ]; do
    while [ $mode -le 3 ]; do
        # mpiexec -n $SLURM_NTASKS $program `printf "${protocol}/iteration_%03d_mode_%d.xml"  ${iter} ${mode}`
        srun -n $SLURM_NTASKS --mpi=pmi2 --export LD_LIBRARY_PATH $program `printf "${protocol}/iteration_%03d_mode_%d.xml"  ${iter} ${mode}`
	rm *.tmp
	if [ $mode -eq 1 ]; then
	# Generate montage of classes
	    ./montage2.sh $mydir `printf "${pattern}_iteration_%03d" ${iter}`
	    text=`printf "Finished classification for iteration %03d on directory %s" ${iter} ${mydir}`
	    if [ -n $mailto ]; then
		echo $text | mutt -s "$text" $mailto
#		echo $text | mutt -a  `printf "${pattern}_iteration_%03d.png" ${iter}` -s "$text" $mailto
	    fi
        # Wait for user intervention
	    while [ ! -e ./refine ]; do
		echo $text
		sleep 60
	    done
	fi
	if [ $mode -eq 2 ]; then
	# Generate montage of refined classes
	    ./montage2.sh $mydir `printf "${pattern}_iteration_%03d_refined" ${iter}`
	    text=`printf "Finished refinement for iteration %03d on directory %s" ${iter} ${mydir}`
	    if [ -n $mailto ]; then
		echo $text | mutt -s "$text" $mailto
#		echo $text | mutt -a  `printf "${pattern}_iteration_%03d_refined.png" ${iter}` -s "$text" $mailto
	    fi
        # Wait for user intervention
	    while [ ! -e ./mra ]; do
		echo $text
		sleep 60
	    done
	fi
	if [ $mode -eq 3 ]; then
	    text=`printf "Finished mra for iteration %03d on directory %s" ${iter} ${mydir}`
	    if [ -n $mailto ]; then
		echo $text | mutt -s "$text" $mailto
	    fi
	    while [ ! -e ./classify ]; do
		echo $text
		sleep 60
	    done
	fi
	if [ ! -e ./repeat ]; then
	    let mode=mode+1
	fi
    done
    let mode=1
    let iter=iter+1
done 
#
# mpdallexit

# nodefil1="$(echo ${PBS_NODEFILE} | awk -F.biobos '{print $1}')"
# nodefil2=${nodefil1##*/}
# mv ~/stavg.e${nodefil2} . && chmod 664 3DAVG.e${nodefil2}
# mv ~/stavg.o${nodefil2} . && chmod 664 3DAVG.o${nodefil2}

