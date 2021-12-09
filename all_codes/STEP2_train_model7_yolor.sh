#!/usr/bin/env bash
cd ..
cd YOLO
cd yolor

bash scripts/get_pretrain.sh
python train.py --batch-size 16 --img-size 576 576 --data ../endoscopy.yaml --cfg cfg/yolor_w6.cfg --device 0 --sync-bn --name yolor_p6 --hyp hyp.scratch.1280.yaml --epochs 400 --weights ckpts --multi-scale
