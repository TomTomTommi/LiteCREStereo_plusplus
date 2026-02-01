#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

torchrun --nproc_per_node=1 train_stereo.py --validation_frequency 100000 \
--ckpt_path ./checkpoints/crestereo++_sceneflow --log_path ./checkpoints/crestereo++_sceneflow  \
--batch_size 24 --train_datasets sceneflow --lr 0.0001 --num_steps 120000
