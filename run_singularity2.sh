#!/bin/bash
CODEDIR=$1
DATADIR=$2
RESULTSDIR=$3
PARAM=$4
cd $CODEDIR

export PYTHONUSERBASE=$SCRATCHDIR
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
pip install -r requirements.txt

python run_vessels.py $DATADIR $RESULTSDIR $PARAM