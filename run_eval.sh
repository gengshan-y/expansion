# evaluate on KITTI val set (ID 0,5,10,...195)
datapath=/ssd/kitti_scene
modelpath=./weights
dataset=2015val
modelname=exp-kitti-train
#modelname=exp-kitti-trainval

CUDA_VISIBLE_DEVICES=0 python submission.py --dataset $dataset --datapath $datapath/training/   --outdir $modelpath/$modelname/ --loadmodel $modelpath/$modelname/$modelname.pth  --testres 1 --fac 2 --maxdisp 512
python eval_exp.py   --path $modelpath/$modelname/  --dataset $dataset
python eval_flow.py       --path $modelpath/$modelname/ --vis no --dataset $dataset
