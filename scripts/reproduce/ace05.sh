export PYTHONPATH="$PWD"
DATA_DIR="/mnt/mrc/ace2005"
BERT_DIR="/mnt/mrc/wwm_uncased_L-24_H-1024_A-16"

BERT_DROPOUT=0.1
MRC_DROPOUT=0.4
LR=1e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0

OUTPUT_DIR="/mnt/mrc/train_logs/ace2005/ace2005_20200917_wwmlarge_sgd_warm${WARMUP}lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_bsz32_gold_span_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}"
mkdir -p $OUTPUT_DIR

python trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 8 \
--gpus="0,1,2,3" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--distributed_backend=ddp \
--val_check_interval 0.25 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $MRC_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs 20 \
--span_loss_candidates "pred_and_gold" \
--weight_span $SPAN_WEIGHT \
--warmup_steps $WARMUP \
--max_length $MAXLEN \
--gradient_clip_val $MAXNORM \
--optimizer "adamw"
