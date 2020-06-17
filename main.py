from __future__ import print_function
import cv2
cv2.setNumThreads(0)
import sys
import pdb
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from utils.flowlib import flow_to_image
from models import *
from utils import logger
torch.backends.cudnn.benchmark=True
from models.VCN_exp  import VCN 
from utils.multiscaleloss import realEPE


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size (default 256)')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size (default 1)')
parser.add_argument('--logname', default='exp-1',
                    help='name of the log file (default exp-1)')
parser.add_argument('--database',
                    help='path to the database (required)')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model (default None)')
parser.add_argument('--loadflow', default=None,
                    help='path of the pre-trained flow model (default None)')
parser.add_argument('--savemodel',
                    help='path to save the model (required)')
parser.add_argument('--retrain', default='true',
                    help='whether to reset moving mean / other hyperparameters (default true)')
parser.add_argument('--stage', default='expansion',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval, expansion, expansion2015train, expansion2015tv} (deafult expansion)')
parser.add_argument('--ngpus', type=int, default=1,
                    help='number of gpus to use (default 1)')
parser.add_argument('--itersave', default='./',
                    help='a dir to save iteration counts (default ./)')
parser.add_argument('--niter', type=int ,default=40000,
                    help='maximum iteration (default 40k)')
args = parser.parse_args()

# fix random seed
torch.manual_seed(1)
def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(8)  # do it again
torch.cuda.manual_seed(1)

## set hyperparameters for training
ngpus = args.ngpus
batch_size = 4*ngpus
if args.stage == 'chairs' or args.stage == 'things':
    lr_schedule = 'slong_ours'
else:
    lr_schedule = 'rob_ours'
baselr = 1e-3
worker_mul = int(2)

if 'expansion' in args.stage:
    datashape = [256,704]
    batch_size = 8*ngpus
    worker_mul = int(1)
elif args.stage == 'chairs' or args.stage == 'things':
    datashape = [320,448]
elif '2015' in args.stage:
    datashape = [256,768]
elif 'sintel' in args.stage:
    datashape = [320,576]
else: 
    print('error')
    exit(0)


## dataloader
## expansion datasets
if 'expansion' in args.stage:
    from dataloader import depthloader as dd
    if '2015' in args.stage:
        if 'train' in args.stage:
            from dataloader import kitti15list_train as lk15
        elif 'tv' in args.stage:
            from dataloader import kitti15list as lk15
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        disp0 = [i.replace('flow_occ','disp_occ_0') for i in flowl0]
        disp1 = [i.replace('flow_occ','disp_occ_1') for i in flowl0]
        calib = [i.replace('flow_occ','calib')[:-7]+'.txt' for i in flowl0]
        loader_kitti15_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0,prob=0.5,sc=True,disp0=disp0, disp1=disp1, calib=calib)
    else:
        from dataloader import sceneflowlist as lsf
        iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Driving/'%args.database, level=6)
        loader_driving_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, sc=True,disp0=disp0,disp1=dispc,calib=calib)
        #iml0, iml1, flowl0, disp0, dispc, calib = lsf.dataloader('%s/Monkaa/'%args.database, level=4)
        #loader_monkaa_sc = dd.myImageFloder(iml0,iml1,flowl0, shape=datashape, sc=True,disp0=disp0,disp1=dispc,calib=calib)
else:
    from dataloader import robloader as dr
    if args.stage == 'chairs' or 'sintel' in args.stage:
        # flying chairs
        from dataloader import chairslist as lc
        iml0, iml1, flowl0 = lc.dataloader('%s/FlyingChairs_release/data/'%args.database)
        with open('order.txt','r') as f:
            order = [int(i) for i in f.readline().split(' ')]
        with open('FlyingChairs_train_val.txt', 'r') as f:
            split = [int(i) for i in f.readlines()]
        iml0 = [iml0[i] for i in order     if split[i]==1]
        iml1 = [iml1[i] for i in order     if split[i]==1]
        flowl0 = [flowl0[i] for i in order if split[i]==1]
        loader_chairs = dr.myImageFloder(iml0,iml1,flowl0, shape = datashape)
    
    
    if args.stage == 'things' or 'sintel' in args.stage:
        # flything things
        from dataloader import thingslist as lt
        iml0, iml1, flowl0 = lt.dataloader('/ssd0/gengshay/FlyingThings3D_subset/train/')
        loader_things = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=1)
    
    # fine-tuning datasets
    if args.stage == '2015train':
        from dataloader import kitti15list_train as lk15
    else:
        from dataloader import kitti15list as lk15
    if args.stage == 'sinteltrain':
        from dataloader import sintellist_train as ls
    else:
        from dataloader import sintellist as ls
    from dataloader import kitti12list as lk12
    from dataloader import hd1klist as lh
    
    if 'sintel' in args.stage:
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, noise=0)  # SINTEL
        iml0, iml1, flowl0 = lh.dataloader('%s/rob_flow/training/'%args.database)
        loader_hd1k = dr.myImageFloder(iml0,iml1,flowl0,shape=datashape, scale=0.5,order=0, noise=0)
        iml0, iml1, flowl0 = ls.dataloader('%s/rob_flow/training/'%args.database)
        loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0)
        #loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0, scale_aug=[0.2,0.])
    if '2015' in args.stage:
        iml0, iml1, flowl0 = lk12.dataloader('%s/data_stereo_flow/training/'%args.database)
        #loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5, scale_aug=[0.2,0.])
        loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)
        iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
        #loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5, scale_aug=[0.2,0.])  # KITTI
        loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)  # KITTI



## aggregate datasets
if 'expansion' in args.stage:
    if '2015' in args.stage:
        data_inuse = torch.utils.data.ConcatDataset([loader_kitti15_sc]*10000)
    else:
        data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200) 
        #data_inuse = torch.utils.data.ConcatDataset([loader_driving_sc]*200+[loader_monkaa_sc]*100)
    for i in data_inuse.datasets:
        i.black = False
        i.cover = True
elif args.stage=='chairs':
    data_inuse = torch.utils.data.ConcatDataset([loader_chairs]*100) 
elif args.stage=='things':
    data_inuse = torch.utils.data.ConcatDataset([loader_things]*100) 
elif '2015' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*50+[loader_kitti12]*50)
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
elif 'sintel' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*200*6+[loader_hd1k]*40*6 + [loader_sintel]*150*6 + [loader_chairs]*2*6 + [loader_things]*6)
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
    baselr = 1e-4
else:
    print('error')
    exit(0)


print('Total iterations: %d'%(len(data_inuse)//batch_size))
print('Max iterations: %d'  %(args.niter))

#TODO
model = VCN([batch_size//ngpus]+data_inuse.datasets[0].shape[::-1], md=[int(4*(args.maxdisp/256)), 4,4,4,4], fac=args.fac)
model = nn.DataParallel(model)
model.cuda()

total_iters = 0
mean_L=[[0.33,0.33,0.33]]
mean_R=[[0.33,0.33,0.33]]
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if args.retrain == 'true':
        print('re-training')
        if 'expansion' in args.stage:
            print('resuming mean from %d'%total_iters)
            mean_L=pretrained_dict['mean_L']
            mean_R=pretrained_dict['mean_R']
    else:
        with open('%s/iter_counts-%d.txt'%(args.itersave, int(args.logname.split('-')[-1])), 'r') as f:
            total_iters = int(f.readline())
        print('resuming from %d'%total_iters)
        mean_L=pretrained_dict['mean_L']
        mean_R=pretrained_dict['mean_R']

if args.loadflow is not None:
    pretrained_dict = torch.load(args.loadflow)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}  # to be compatible with prior models
    #pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'f_modules' in k or 'p_modules' in k or 'oor_modules' in k or 'fuse_modules' in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), amsgrad=False)

def train(imgL,imgR,flowl0,imgAux,intr, imgoL, imgoR, occp):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        flowl0 = Variable(torch.FloatTensor(flowl0))

        imgL, imgR, flowl0 = imgL.cuda(), imgR.cuda(), flowl0.cuda()
        mask = (flowl0[:,:,:,2] == 1) & (flowl0[:,:,:,0].abs() < args.maxdisp) & (flowl0[:,:,:,1].abs() < (args.maxdisp//args.fac))
        if not imgAux is None:
            imgAux = imgAux.cuda()
            imgoL, imgoR = imgoL.float().cuda(), imgoR.float().cuda()
            mask = mask & (imgAux[:,:,:,0] < 100) & (imgAux[:,:,:,0] > 0.01)  # depth, d1,d2,d2,flow3d
            exp_flag = True
        else:
            exp_flag = False
        mask.detach_(); 


        # rearrange inputs
        groups = []
        for i in range(ngpus):
            groups.append(imgL[i*batch_size//ngpus:(i+1)*batch_size//ngpus])
            groups.append(imgR[i*batch_size//ngpus:(i+1)*batch_size//ngpus])

        # forward-backward
        optimizer.zero_grad()
        output = model(torch.cat(groups,0), [flowl0,mask,imgAux,intr, imgoL, imgoR, occp, exp_flag])
        loss = output[-3].mean()
        loss.backward()
        optimizer.step()

        if np.isnan(np.asarray(model.module.dc2_conv7.weight.max().detach().cpu())):
            pdb.set_trace()
            output = modela(torch.cat([imgL,imgR],0))

        vis = {}
        vis['output2'] = output[0].detach().cpu().numpy()
        vis['output3'] = output[1].detach().cpu().numpy()
        vis['output4'] = output[2].detach().cpu().numpy()
        vis['output5'] = output[3].detach().cpu().numpy()
        vis['output6'] = output[4].detach().cpu().numpy()
        vis['mid'] = output[6][0].detach().cpu().numpy()
        vis['exp'] = output[7][0].detach().cpu().numpy()
        vis['gt'] = flowl0[:,:,:,:].detach().cpu().numpy()
        if mask.sum():
            vis['AEPE'] = realEPE(output[0].detach(), flowl0.permute(0,3,1,2).detach(),mask,sparse=False)
        vis['mask'] = mask
        return loss.data,vis

def adjust_learning_rate(optimizer, total_iters):
    if lr_schedule == 'slong':
        if total_iters < 200000:
            lr = baselr
        elif total_iters < 300000:
            lr = baselr/2.
        elif total_iters < 400000:
            lr = baselr/4.
        elif total_iters < 500000:
            lr = baselr/8.
        elif total_iters < 600000:
            lr = baselr/16.
    if lr_schedule == 'slong_ours':
        if total_iters < 70000:
            lr = baselr
        elif total_iters < 130000:
            lr = baselr/2.
        elif total_iters < 190000:
            lr = baselr/4.
        elif total_iters < 240000:
            lr = baselr/8.
        elif total_iters < 290000:
            lr = baselr/16.
    if lr_schedule == 'slong_pwc':
        if total_iters < 400000:
            lr = baselr
        elif total_iters < 600000:
            lr = baselr/2.
        elif total_iters < 800000:
            lr = baselr/4.
        elif total_iters < 1000000:
            lr = baselr/8.
        elif total_iters < 1200000:
            lr = baselr/16.
    if lr_schedule == 'sfine_pwc':
        if total_iters < 1400000:
            lr = baselr/10.
        elif total_iters < 1500000:
            lr = baselr/20.
        elif total_iters < 1600000:
            lr = baselr/40.
        elif total_iters < 1700000:
            lr = baselr/80.
    if lr_schedule == 'sfine':
        if total_iters < 700000:
            lr = baselr/10.
        elif total_iters < 750000:
            lr = baselr/20.
        elif total_iters < 800000:
            lr = baselr/40.
        elif total_iters < 850000:
            lr = baselr/80.
    if lr_schedule == 'rob_ours':
        if total_iters < 30000:
            lr = baselr
        elif total_iters < 40000:
            lr = baselr / 2.
        elif total_iters < 50000:
            lr = baselr / 4.
        elif total_iters < 60000:
            lr = baselr / 8.
        elif total_iters < 70000:
            lr = baselr / 16.
        elif total_iters < 100000:
            lr = baselr
        elif total_iters < 110000:
            lr = baselr / 2.
        elif total_iters < 120000:
            lr = baselr / 4.
        elif total_iters < 130000:
            lr = baselr / 8.
        elif total_iters < 140000:
            lr = baselr / 16.
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# get global counts                
with open('%s/iter_counts-%d.txt'%(args.itersave, int(args.logname.split('-')[-1])), 'w') as f:
    f.write('%d'%total_iters)

def main():
    TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, shuffle= True, num_workers=int(worker_mul*batch_size), drop_last=True, worker_init_fn=_init_fn, pin_memory=True)
    log = logger.Logger(args.savemodel, name=args.logname)
    start_full_time = time.time()
    global total_iters

    # training loop
    for batch_idx, databatch in enumerate(TrainImgLoader):
        if batch_idx > args.niter: break
        if 'expansion' in args.stage:
            imgL_crop, imgR_crop, flowl0,imgAux,intr, imgoL, imgoR, occp  = databatch
        else:
            imgL_crop, imgR_crop, flowl0 = databatch
            imgAux,intr, imgoL, imgoR, occp = None,None,None,None,None
        if batch_idx % 100 == 0:
            adjust_learning_rate(optimizer,total_iters)
        if total_iters < 1000 and not 'expansion' in args.stage:
            # subtract mean
            mean_L.append( np.asarray(imgL_crop.mean(0).mean(1).mean(1)) )
            mean_R.append( np.asarray(imgR_crop.mean(0).mean(1).mean(1)) )
        imgL_crop -= torch.from_numpy(np.asarray(mean_L).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()
        imgR_crop -= torch.from_numpy(np.asarray(mean_R).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()

        start_time = time.time() 
        loss,vis = train(imgL_crop,imgR_crop, flowl0, imgAux,intr, imgoL, imgoR, occp)
        print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))

        if total_iters %10 == 0:
            log.scalar_summary('train/loss_batch',loss, total_iters)
            log.scalar_summary('train/aepe_batch',vis['AEPE'], total_iters)
        if total_iters %100 == 0:
            log.image_summary('train/left',imgL_crop[0:1],total_iters)
            log.image_summary('train/right',imgR_crop[0:1],total_iters)
            if len(np.asarray(vis['gt']))>0:
                log.histo_summary('train/gt_hist',np.asarray(vis['gt']).reshape(-1,3)[np.asarray(vis['gt'])[:,:,:,-1].flatten().astype(bool)][:,:2], total_iters)
            gu = vis['gt'][0,:,:,0]; gv = vis['gt'][0,:,:,1]
            gu = gu*np.asarray(vis['mask'][0].float().cpu());  gv = gv*np.asarray(vis['mask'][0].float().cpu())
            mask = vis['mask'][0].float().cpu()
            log.image_summary('train/gt0', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],total_iters)
            log.image_summary('train/output2',flow_to_image(vis['output2'][0].transpose((1,2,0)))[np.newaxis],total_iters)
            log.image_summary('train/output3',flow_to_image(vis['output3'][0].transpose((1,2,0)))[np.newaxis],total_iters)
            log.image_summary('train/output4',flow_to_image(vis['output4'][0].transpose((1,2,0)))[np.newaxis],total_iters)
            log.image_summary('train/output5',flow_to_image(vis['output5'][0].transpose((1,2,0)))[np.newaxis],total_iters)
            log.image_summary('train/output6',flow_to_image(vis['output6'][0].transpose((1,2,0)))[np.newaxis],total_iters)
            if 'expansion' in args.stage:
                log.image_summary('train/mid_gt',(1+imgAux[:1,:,:,6]/imgAux[:1,:,:,0]).log() ,total_iters)
                log.image_summary('train/mid',vis['mid'][np.newaxis],total_iters)
                log.image_summary('train/exp',vis['exp'][np.newaxis],total_iters)
            torch.cuda.empty_cache()
        total_iters += 1
        # get global counts                
        with open('%s/iter_counts-%d.txt'%(args.itersave,int(args.logname.split('-')[-1])), 'w') as f:
            f.write('%d'%total_iters)

        if (total_iters + 1)%2000==0:
            #SAVE
            savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.pth'
            save_dict = model.state_dict()
            save_dict = collections.OrderedDict({k:v for k,v in save_dict.items() if ('reg_modules' not in k or 'conv1' in k) and ('grid' not in k) and ('flow_reg' not in k)})
            torch.save({
                'iters': total_iters,
                'state_dict': save_dict,
                'mean_L': mean_L,
                'mean_R': mean_R,
            }, savefilename)
        
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)


if __name__ == '__main__':
    main()
