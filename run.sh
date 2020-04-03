#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-2}"
density="${density:-1}"
compressor="${compressor:-bucket}"
nwpernode=1
nstepsupdate=1
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
GRADSPATH=./logs/iclr

mpirun -np $nworkers --host inv20ib,inv21ib -bind-to none -map-by slot \
    -x UCX_TLS=rc,sm,cuda_copy,cuda_ipc --mca orte_base_help_aggregate 0 \
    -x NCCL_DEBUG=WARN -x LD_LIBRARY_PATH -x PATH -x DEBUG=1 \
    $PY dist_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --saved-dir $GRADSPATH 
