#!/bin/bash
#for model in ddgan_sd_v10 ddgan_laion2b_v2 ddgan_ddb_v1 ddgan_ddb_v2 ddgan_ddb_v3 ddgan_ddb_v4;do 
#for model in ddgan_ddb_v2 ddgan_ddb_v3 ddgan_ddb_v4 ddgan_ddb_v5;do 
#for model in ddgan_ddb_v4 ddgan_ddb_v6 ddgan_ddb_v7 ddgan_laion_aesthetic_v15;do 
#for model in ddgan_ddb_v6;do 
for model in ddgan_laion_aesthetic_v15;do 
    if [ "$model" == "ddgan_ddb_v3" ]; then
        bs=32
    elif [ "$model" == "ddgan_laion_aesthetic_v15" ]; then
        bs=32
    elif [ "$model" == "ddgan_ddb_v6" ]; then
        bs=32
    elif [ "$model" == "ddgan_ddb_v4" ]; then
        bs=16
    else
        bs=64
    fi
    sbatch --partition dc-gpu -t 360 -N 1 -n1 scripts/run_jurecadc_ddp.sh run.py test $model --cond-text=parti_prompts.txt  --batch-size=$bs --epoch=-1 --compute-clip-score --eval-name=parti;
    sbatch --partition dc-gpu -t 360 -N 1 -n1 scripts/run_jurecadc_ddp.sh run.py test $model --fid --real-img-dir inception_statistics_coco_val2014_256x256.npz --cond-text coco_val2014_captions.txt  --batch-size=$bs --epoch=-1 --nb-images-for-fid=30000 --eval-name=coco --compute-clip-score;
    sbatch --partition dc-gpu -t 360 -N 1 -n1 scripts/run_jurecadc_ddp.sh run.py test $model --cond-text=drawbench.txt  --batch-size=$bs --epoch=-1 --compute-clip-score --eval-name=drawbench;
done
