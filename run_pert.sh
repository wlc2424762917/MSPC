#! /bin/bash

source /ocean/projects/asc170022p/yanwuxu/miniconda/etc/profile.d/conda.sh
conda activate crossMoDA

direction='AtoB'
dataroot='../data/cityscapes'
batch_size=12
load_size=128
crop_size=128

netG=resnet_9blocks
netD=basic

python train.py --dataroot $dataroot --model maxpert_gan \
--pool_size 50 --no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size --identity 0.5 --lambda_pert 0.1 \
--direction $direction;
python test.py --dataroot $dataroot --model maxpert_gan --eval \
--no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size --identity 0.5 --lambda_pert 0.1 \
--direction $direction

python train.py --dataroot $dataroot --model $model --gan_mode lsgan \
--bounded $bounded --grid_size 2 --pert_threshold $pert_threshold --lambda_blank $lambda_blank \
--pool_size 50 --no_dropout --load_size $load_size --crop_size $crop_size \
--netG $netG --netD $netD --batch_size $batch_size --identity $identity \
--direction $direction;

python train.py --dataroot './data/face_unaligned/face_unaligned' --model maxgcpert3_gan --pool_size 50 --no_dropout --load_size 128 --crop_size 128 --netG resnet_9blocks --netD basic --batch_size 1 --identity 0.5 --lambda_pert 0.1 --direction 'AtoB'