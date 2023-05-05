CUDA_VISIBLE_DEVICES=0 python3 train.py --batchSize 10 \
  --dataRoot 'datasets/images' \
  --net 'idr' \
  --mode 'l1' \
  --lr 2e-4 \
  --lr_decay_iters 50 \
  --loadSize 512 \
  --modelsSavePath './ckpts/' \
  --logPath 'logs'