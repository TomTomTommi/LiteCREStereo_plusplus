#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python evaluate_stereo.py --restore_ckpt ./checkpoints/crestereo_sf.pth --dataset middlebury_H
python evaluate_stereo.py --restore_ckpt ./checkpoints/crestereo_sf.pth --dataset eth3d
python evaluate_stereo.py --restore_ckpt ./checkpoints/crestereo_sf.pth --dataset kitti
#python evaluate_stereo.py --restore_ckpt ./checkpoints/crestereo_sf.pth --dataset middlebury_F
#python evaluate_stereo.py --restore_ckpt ./checkpoints/crestereo_sf.pth --dataset middlebury_Q
