#!/bin/bash
DATADIR=$1
cd $SCRATCHDIR/DeepRetinaSegmentation

# export PYTHONUSERBASE=$SCRATCHDIR
# export PATH=$PYTHONUSERBASE/bin:$PATH
# export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
# pip install SimpleITK


python main_vessels.py $DATADIR
