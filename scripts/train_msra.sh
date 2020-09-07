export PYTHONPATH="$PWD"

DATA_DIR="/mnt/mrc/zh_msra"
BERT_DIR="/mnt/mrc/chinese_L-12_H-768_A-12"
OUTPUT_DIR="/mnt/mrc/train_logs/zh_msra_maxlen128_lr2e-5"

python trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length 128 \
--batch_size 16 \
--gpus="0,1,2,3" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr 2e-5 \
--workers 4 \
--distributed_backend=ddp \
--val_check_interval 1.0 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout 0.2 \
--max_epochs 30 \
--pretrained_checkpoint "/mnt/mrc/train_logs/zh_msra_lr2e-5/epoch=9.ckpt"


## Grid Search
export PYTHONPATH="$PWD"

DATA_DIR="/mnt/mrc/zh_msra"
BERT_DIR="/mnt/mrc/chinese_L-12_H-768_A-12"

for lr in 6e-6 2e-5 1e-5 8e-6; do
  OUTPUT_DIR="/mnt/mrc/train_logs/zh_msra_lr${lr}"
  LOGFILE=$OUTPUT_DIR/log.txt
  mkdir -p $OUTPUT_DIR
  python trainer.py \
    --data_dir $DATA_DIR \
    --bert_config_dir $BERT_DIR \
    --max_length 128 \
    --batch_size 16 \
    --gpus="0,1,2,3" \
    --precision=16 \
    --progress_bar_refresh_rate 1 \
    --lr $lr \
    --workers 4 \
    --distributed_backend=ddp \
    --val_check_interval 0.25 \
    --accumulate_grad_batches 1 \
    --default_root_dir $OUTPUT_DIR \
    --mrc_dropout 0.2 \
    --max_epochs 15 \
  >$LOGFILE
done
