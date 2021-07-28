REPO_PATH=/userhome/xiaoya/github_mrc
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
DATA_DIR=/userhome/xiaoya/dataset/ace2005
BERT_DIR=/userhome/xiaoya/bert/uncased_L-24_H-1024_A-16

BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=2e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0

OUTPUT_DIR="/userhome/xiaoya/outputs/mrc-ner/ace2005/large_${WARMUP}lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}"
mkdir -p $OUTPUT_DIR

nohup python ${REPO_PATH}/trainer.py \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 8 \
--gpus="0,1,2,3" \
--precision=16 \
--progress_bar_refresh_rate 2 \
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
--optimizer "adamw" > ${OUTPUT_DIR}/train_log.txt & tail -f ${OUTPUT_DIR}/train_log.txt