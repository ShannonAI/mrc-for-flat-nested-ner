export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
DATA_DIR="/mnt/mrc/zh_onto4"
BERT_DIR="/mnt/mrc/chinese_roberta_wwm_large_ext_pytorch"

# todo(yuxian): warmup ratio

MAXLENGTH=128
WEIGHT_SPAN=0.1
lr=8e-6
OUTPUT_DIR="/mnt/mrc/train_logs/zh_onto/zh_onto_20200914_lr${lr}_maxlen${MAXLENGTH}_newdataset_weight_span${WEIGHT_SPAN}"
mkdir -p $OUTPUT_DIR

python trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLENGTH \
--batch_size 4 \
--gpus="0,1,2,3" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $lr \
--workers 0 \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--max_epochs 20 \
--chinese \
--span_loss_candidates "pred_and_gold" \
--weight_span $WEIGHT_SPAN \
--mrc_dropout 0.2 \
--warmup_steps 0 \
--gradient_clip_val 1.0

