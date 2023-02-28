#! /bin/bash


pert_threshold=2.0 ###scale of perturbation 
lambda_blank=50.0  ###constraint coefficient of perturbation
identity=10.0
direction='AtoB'

dataroot='./data/face_unaligned/face_unaligned'
batch_size=4
load_size=144
crop_size=128
model=maxgcpert3_gan
bounded=unbounded
netG=resnet_6blocks
netD=basic

python train.py --dataroot $dataroot --model $model --gan_mode lsgan \
--bounded $bounded --grid_size 2 --pert_threshold $pert_threshold --lambda_blank $lambda_blank \
--pool_size 50 --no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size --identity $identity \
--direction $direction;
#python test.py --dataroot $dataroot --model $model --eval \
#--bounded $bounded --grid_size 2 --pert_threshold $pert_threshold --lambda_blank $lambda_blank \
#--no_dropout --load_size $load_size --crop_size $crop_size \
#--netG $netG --netD $netD --batch_size $batch_size --identity $identity \
#--direction $direction

python train.py --dataroot './data/face_unaligned/face_unaligned' --model maxgcpert3_gan --gan_mode lsgan --bounded unbounded --pool_size 50 --no_dropout --load_size 128 --crop_size 128 --netG resnet_9blocks --netD basic --batch_size 1 --identity 0.5 --direction 'AtoB'
python test.py --dataroot './data/face_unaligned/face_unaligned' --model maxgcpert3_gan --eval --bounded unbounded --pool_size 50 --no_dropout --load_size 128 --crop_size 128 --netG resnet_9blocks --netD basic --batch_size 1 --identity 0.5 --direction 'AtoB'