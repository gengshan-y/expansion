# evaluate on KITTI test set (ID 0,1,2,...199)
datapath=/ssd/kitti_scene
modelpath=./weights
dataset=2015test
modelname=exp-kitti-trainval

CUDA_VISIBLE_DEVICES=0 python submission.py --dataset $dataset --datapath $datapath/testing/   --outdir $modelpath/$modelname/ --loadmodel $modelpath/$modelname/$modelname.pth  --testres 1 --fac 2 --maxdisp 512
