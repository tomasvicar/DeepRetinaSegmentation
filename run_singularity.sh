#!/bin/bash
DATADIR=$1
cd $SCRATCHDIR/DeepRetinaSegmentation_tmp/DeepRetinaSegmentation/

export PYTHONUSERBASE=$SCRATCHDIR
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
pip install nvidia-ml-py3==7.352.0
pip install segmentation-models-pytorch

python main_vessels.py $DATADIR