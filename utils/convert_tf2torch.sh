# convert tf model to pytorch format

export BERT_BASE_DIR=/mnt/mrc/wwm_uncased_L-24_H-1024_A-16

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/model.ckpt \
  --config $BERT_BASE_DIR/config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin
