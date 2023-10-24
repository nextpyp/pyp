#!/bin/bash

# parse fields from config file
parse () {
    config=$1
    name=$2
    line=`cat $config | grep $name | grep -v "#"`
    IFS='=' read -r -a array <<< "$line"
    value=`echo ${array[1]} | sed -e 's/^"//' -e 's/"$//'`
    echo $value
}

echo "launching..."

# retrieve configuration file
binDir=`dirname $0`
configFile=${PYP_CONFIG}
if [[ ! -z ${#configFile} && ! -f $configFile ]]
then
    configFile=${HOME}/.pyp/config.toml
fi
        
if [ ! -f $configFile ]
then
    echo -e "\e[31mERROR\e[0m - Please create a PYP configuration file in ~/.pyp/config.toml or set $PYP_CONFIG to the location of your config.toml."
    exit 1
fi

# resolve pyp command and parameters
name=`basename "$0"`
pypCommand=`echo /opt/pyp/bin/run/$name`
pypParameters=$@

if [ ! -z "${SINGULARITY_CONTAINER}" ]
then
    command="$pypCommand $pypParameters"
    # launch
    eval $command
else

    # retrieve container location
    pypContainer="$(parse $configFile "container")"

    if [ ! -f $pypContainer ]
    then
        echo -e "\e[31mERROR\e[0m - Can't find singularity container $pypContainer. Please check configuration file $configFile"
        exit 1
    fi

    # singularity binds
    singularityBinds="$(parse $configFile "binds")"

    if [ ! -z ${#singularityBinds} ]
    then
        singularityBinds=`echo $singularityBinds | sed -e "s/\[//" | sed -e "s|\]||" | sed -e "s|\'||" | sed -e "s|\"||"  | sed 's/ //g'`
        singularityBinds="-B "$singularityBinds
    else
        singularityBinds=""
    fi

    # local scratch
    pypScratch="$(parse $configFile "scratch")"
    if [ -d ${#pypScratch} ]
    then
        singularityBinds="-B "$pypScratch
    else
        IFS='$' read -r -a array <<< "$pypScratch"
        pypScratch=`echo ${array[0]} | sed -e 's/^"//' -e 's/"$//'`
        if [ -d ${#pypScratch} ]
        then
            singularityBinds="-B "$pypScratch
        fi
    fi

    singularityBinds=`echo ${singularityBinds} -B ${pypScratch}`

    # add .ssh to binds
    mkdir -p ${HOME}/.config
    mkdir -p ${HOME}/.cache
    singularityBinds=`echo ${singularityBinds} --no-home -B ${HOME}/.ssh -B ${HOME}/.config -B ${HOME}/.cache`

    # pyp sources
    pypLocation="$(parse $configFile "sources")"

    if [ -d ${pypLocation} ]
    then
        singularityBinds=`echo ${singularityBinds} -B ${pypLocation}:/opt/pyp`
    fi

    # singularity
    singularityExecutable="$(parse $configFile "singularity")"
    length=`echo ${#singularityExecutable}`
    if [ $length -gt 3 ]
    then
        singularityExecutable="$singularityExecutable > /dev/null 2>&1; `which singularity`"
    else
        singularityExecutable="`which singularity`"
    fi

    # final command
    command="$singularityExecutable --quiet --silent exec --pwd $PWD $singularityBinds $pypContainer $pypCommand $pypParameters"
    # launch
    eval $command
fi
