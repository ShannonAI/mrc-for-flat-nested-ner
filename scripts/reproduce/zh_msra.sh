export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
DATA_DIR="/mnt/mrc/zh_msra"
BERT_DIR="/mnt/mrc/chinese_roberta_wwm_large_ext_pytorch"
SPAN_WEIGHT=0.1
DROPOUT=0.2
LR=8e-6
MAXLEN=128

OUTPUT_DIR="/mnt/mrc/train_logs/zh_msra/zh_msra_bertlarge_lr${LR}20200913_dropout${DROPOUT}_bsz16_maxlen${MAXLEN}"

mkdir -p $OUTPUT_DIR

python trainer.py \
--chinese \
--data_dir $DATA_DIR \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 4 \
--gpus="0,1,2,3" \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--distributed_backend=ddp \
--val_check_interval 0.5 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $DROPOUT \
--max_epochs 20 \
--weight_span $SPAN_WEIGHT \
--span_loss_candidates "pred_and_gold"
