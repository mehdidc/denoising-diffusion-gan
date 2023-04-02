ml purge
ml use $OTHERSTAGES
ml Stages/2022
ml GCC/11.2.0
ml OpenMPI/4.1.2
ml CUDA/11.5
ml cuDNN/8.3.1.22-CUDA-11.5
ml NCCL/2.12.7-1-CUDA-11.5
ml PyTorch/1.11-CUDA-11.5
ml Horovod/0.24
ml torchvision/0.12.0
source /p/home/jusers/cherti1/jureca/ccstdl/code/feed_forward_vqgan_clip/envs/jureca_2022/bin/activate
export HOROVOD_CACHE_CAPACITY=4096
export CUDA_VISIBLE_DEVICES=0,1,2,3
