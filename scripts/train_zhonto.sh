export PYTHONPATH="$PWD"
# todo(yuxian: note 之前跑的是256的), 需要根据 weight_sum的变化，lr*3
DATA_DIR="/mnt/mrc/zh_onto4"
BERT_DIR="/mnt/mrc/chinese_L-12_H-768_A-12"


for lr in 8e-5 7e-5 6e-5 4e-5 2e-5 1e-5; do
  OUTPUT_DIR="/mnt/mrc/train_logs/zh_onto/zh_onto_20200909_gridsearch_lr${lr}"

  mkdir -p $OUTPUT_DIR

  python trainer.py \
  --data_dir $DATA_DIR \
  --bert_config_dir $BERT_DIR \
  --max_length 256 \
  --batch_size 3 \
  --gpus="0,1,2,3" \
  --precision=16 \
  --progress_bar_refresh_rate 1 \
  --lr $lr \
  --workers 4 \
  --distributed_backend=ddp \
  --val_check_interval 0.5 \
  --accumulate_grad_batches 1 \
  --default_root_dir $OUTPUT_DIR \
  --mrc_dropout 0.2 \
  --max_epochs 40 \
  --chinese
done
