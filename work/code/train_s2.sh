CUDA_VISIBLE_DEVICES=0 python3 train.py --batchSize 8 \
  --dataRoot 'datasets/train_datasets/images' \
  --net 'idr_psnr' \
  --mode 'psnr' \
  --lr 2e-5 \
  --lr_decay_iters 350000 \
  --loadSize 512 \
  --pretrained './ckpts/idr/model_best.pdparams' \
  --modelsSavePath './ckpts/' \
  --logPath 'logs'